#!/usr/bin/env python
"""
Visualize value function predictions on complete episodes.

Generates a video/gif showing:
  - Top: video frames from the episode (e.g. main camera + wrist camera)
  - Bottom: value curve over time, with a moving cursor and color-coded regions

Similar to: https://media.githubusercontent.com/media/MINT-SJTU/Evo-RL/refs/heads/main/website/assets/gifs/value_success.gif

Usage:
    # Visualize specific episode
    python visualize_value_function.py \
        --value_checkpoint checkpoints/value_function/checkpoint_005000/value_function.pt \
        --dataset_root /VLA-Data/scripts/lianqing/data/piper_dataset/task02-put-the-crushed-can-into-the-trash-bin-2026-03-20-30-P1 \
        --episode_index 0 \
        --output_dir vis_output/

    # Visualize multiple episodes
    python visualize_value_function.py \
        --value_checkpoint checkpoints/value_function/checkpoint_005000/value_function.pt \
        --dataset_root /VLA-Data/scripts/lianqing/data/piper_dataset/task02-put-the-crushed-can-into-the-trash-bin-2026-03-20-30-P1 \
        --episode_index 0 1 2 \
        --output_dir vis_output/

    # Also save a static PNG summary
    python visualize_value_function.py \
        --value_checkpoint checkpoints/value_function/checkpoint_005000/value_function.pt \
        --dataset_root /VLA-Data/scripts/lianqing/data/piper_dataset/task02-put-the-crushed-can-into-the-trash-bin-2026-03-20-30-P1 \
        --episode_index 0 \
        --output_dir vis_output/ --save_png
"""

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Color scheme
CMAP_VALUE = LinearSegmentedColormap.from_list(
    "value_cmap", ["#d32f2f", "#ff9800", "#fdd835", "#66bb6a", "#2e7d32"]
)
COLOR_BG = "#1a1a2e"
COLOR_PANEL = "#16213e"
COLOR_TEXT = "#e0e0e0"
COLOR_CURSOR = "#00bcd4"
COLOR_LINE = "#4fc3f7"
COLOR_TARGET = "#81c784"
COLOR_FILL_GOOD = "#2e7d3240"
COLOR_FILL_BAD = "#d32f2f30"


def load_value_function(checkpoint_path: str, device: str = "cuda"):
    from lerobot.policies.pi05.value_function_pi05 import PI05ValueFunction

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = PI05ValueFunction(
        vlm_variant=config["vlm_variant"],
        image_resolution=(224, 224),
        precision=config.get("precision", "float32"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, config, ckpt["step"]


def collect_episode_data(model, dataset, episode_index, image_keys, tokenizer, device):
    """Run value function on every frame of an episode. Returns arrays."""
    ep_data = dataset.meta.episodes[episode_index]
    ep_length = ep_data["length"]
    from_idx = ep_data["dataset_from_index"]

    task_name = ""
    if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
        task_name = str(dataset.meta.tasks.index[0])

    pred_values = []
    target_values = []
    frames_main = []
    frames_wrist = []

    logger.info(f"Episode {episode_index}: {ep_length} frames, task='{task_name}'")

    for i in range(ep_length):
        global_idx = from_idx + i
        sample = dataset[global_idx]

        # Collect raw frames for visualization (before preprocessing)
        for key in image_keys:
            img_tensor = sample[key]  # [C, H, W] float32 in [0, 1]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if "main" in key:
                frames_main.append(img_np)
            elif "wrist" in key:
                frames_wrist.append(img_np)

        # Prepare batch (add batch dim)
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        images, img_masks = model.preprocess_images(batch, image_keys)

        tasks = [task_name]
        encoded = tokenizer(
            tasks, padding="max_length", max_length=200, truncation=True, return_tensors="pt",
        )
        tokens = encoded["input_ids"].to(device)
        masks = encoded["attention_mask"].to(device).bool()

        with torch.no_grad():
            value = model.predict_value(images, img_masks, tokens, masks)
        pred_values.append(value.item())

        # Ground truth: normalized return for successful demo
        steps_remaining = ep_length - 1 - i
        target_values.append(-steps_remaining / ep_length)

        if (i + 1) % 100 == 0:
            logger.info(f"  frame {i+1}/{ep_length}")

    return {
        "episode_index": episode_index,
        "task": task_name,
        "length": ep_length,
        "pred_values": np.array(pred_values),
        "target_values": np.array(target_values),
        "frames_main": frames_main,
        "frames_wrist": frames_wrist,
    }


def render_frame(
    ep_data: dict,
    frame_idx: int,
    fig_width: float = 12,
    fig_height: float = 7,
    dpi: int = 100,
) -> np.ndarray:
    """Render a single visualization frame combining video + value curve."""
    pred = ep_data["pred_values"]
    target = ep_data["target_values"]
    T = ep_data["length"]
    timesteps = np.arange(T)
    time_sec = timesteps / 30.0  # assuming 30 fps

    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.25, wspace=0.05,
                           left=0.05, right=0.95, top=0.92, bottom=0.08)

    # ── Title ─────────────────────────────────────────────────────────
    task_str = ep_data["task"][:60]
    fig.suptitle(
        f"Episode {ep_data['episode_index']}  |  {task_str}  |  frame {frame_idx}/{T}",
        fontsize=12, color=COLOR_TEXT, fontweight="bold",
    )

    # ── Top left: main camera ─────────────────────────────────────────
    ax_main = fig.add_subplot(gs[0, 0])
    ax_main.set_facecolor("black")
    if ep_data["frames_main"]:
        ax_main.imshow(ep_data["frames_main"][frame_idx])
    ax_main.set_title("Main Camera", fontsize=10, color=COLOR_TEXT, pad=4)
    ax_main.axis("off")

    # ── Top right: wrist camera ───────────────────────────────────────
    ax_wrist = fig.add_subplot(gs[0, 1])
    ax_wrist.set_facecolor("black")
    if ep_data["frames_wrist"]:
        ax_wrist.imshow(ep_data["frames_wrist"][frame_idx])
    ax_wrist.set_title("Wrist Camera", fontsize=10, color=COLOR_TEXT, pad=4)
    ax_wrist.axis("off")

    # ── Bottom: value curve ───────────────────────────────────────────
    ax_val = fig.add_subplot(gs[1, :])
    ax_val.set_facecolor(COLOR_PANEL)

    # Color the background based on value (green = high, red = low)
    for i in range(min(frame_idx + 1, T - 1)):
        v = (pred[i] + 1.0)  # map [-1, 0] → [0, 1]
        color = CMAP_VALUE(np.clip(v, 0, 1))
        ax_val.axvspan(time_sec[i], time_sec[min(i + 1, T - 1)], alpha=0.15, color=color, linewidth=0)

    # Plot target line (ground truth)
    ax_val.plot(time_sec, target, color=COLOR_TARGET, linewidth=1.5, alpha=0.5,
                linestyle="--", label="Ground Truth V(t)")

    # Plot predicted value (only up to current frame for animation effect)
    ax_val.plot(time_sec[:frame_idx + 1], pred[:frame_idx + 1],
                color=COLOR_LINE, linewidth=2.5, label="Predicted V(t)")

    # Cursor at current frame
    ax_val.axvline(x=time_sec[frame_idx], color=COLOR_CURSOR, linewidth=2, alpha=0.9)
    ax_val.scatter([time_sec[frame_idx]], [pred[frame_idx]], color=COLOR_CURSOR,
                   s=80, zorder=5, edgecolors="white", linewidths=1.5)

    # Value annotation
    ax_val.annotate(
        f"V = {pred[frame_idx]:.3f}",
        xy=(time_sec[frame_idx], pred[frame_idx]),
        xytext=(15, 15), textcoords="offset points",
        fontsize=11, color=COLOR_CURSOR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLOR_CURSOR, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOR_PANEL, edgecolor=COLOR_CURSOR, alpha=0.9),
    )

    ax_val.set_xlim(-0.5, time_sec[-1] + 0.5)
    ax_val.set_ylim(-1.05, 0.05)
    ax_val.set_xlabel("Time (s)", fontsize=10, color=COLOR_TEXT)
    ax_val.set_ylabel("Value", fontsize=10, color=COLOR_TEXT)
    ax_val.legend(loc="lower right", fontsize=9, facecolor=COLOR_PANEL,
                  edgecolor="#555", labelcolor=COLOR_TEXT)
    ax_val.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax_val.spines["top"].set_visible(False)
    ax_val.spines["right"].set_visible(False)
    ax_val.spines["bottom"].set_color("#555")
    ax_val.spines["left"].set_color("#555")
    ax_val.grid(True, alpha=0.15, color="#555")

    # Render to numpy
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def save_static_summary(ep_data: dict, output_path: str, dpi: int = 150):
    """Save a static PNG showing the full value curve + key frames."""
    pred = ep_data["pred_values"]
    target = ep_data["target_values"]
    T = ep_data["length"]
    timesteps = np.arange(T)
    time_sec = timesteps / 30.0

    # Pick 6 evenly spaced keyframes
    keyframe_indices = np.linspace(0, T - 1, 6, dtype=int)

    fig = plt.figure(figsize=(14, 8), facecolor=COLOR_BG)
    gs = gridspec.GridSpec(3, 6, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.1,
                           left=0.06, right=0.96, top=0.92, bottom=0.08)

    fig.suptitle(
        f"Episode {ep_data['episode_index']}  |  {ep_data['task'][:60]}  |  {T} frames",
        fontsize=13, color=COLOR_TEXT, fontweight="bold",
    )

    # ── Keyframes row 1 (main camera) ─────────────────────────────────
    for col, ki in enumerate(keyframe_indices):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("black")
        if ep_data["frames_main"]:
            ax.imshow(ep_data["frames_main"][ki])
        ax.set_title(f"t={time_sec[ki]:.1f}s", fontsize=8, color=COLOR_TEXT, pad=2)
        ax.axis("off")

    # ── Keyframes row 2 (wrist camera) ────────────────────────────────
    for col, ki in enumerate(keyframe_indices):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor("black")
        if ep_data["frames_wrist"]:
            ax.imshow(ep_data["frames_wrist"][ki])
        ax.axis("off")

    # ── Value curve ───────────────────────────────────────────────────
    ax_val = fig.add_subplot(gs[2, :])
    ax_val.set_facecolor(COLOR_PANEL)

    # Background color fill
    for i in range(T - 1):
        v = pred[i] + 1.0
        color = CMAP_VALUE(np.clip(v, 0, 1))
        ax_val.axvspan(time_sec[i], time_sec[i + 1], alpha=0.12, color=color, linewidth=0)

    ax_val.plot(time_sec, target, color=COLOR_TARGET, linewidth=1.5, alpha=0.6,
                linestyle="--", label="Ground Truth V(t)")
    ax_val.plot(time_sec, pred, color=COLOR_LINE, linewidth=2.5, label="Predicted V(t)")

    # Mark keyframes
    for ki in keyframe_indices:
        ax_val.axvline(x=time_sec[ki], color="#ffffff30", linewidth=1, linestyle=":")
        ax_val.scatter([time_sec[ki]], [pred[ki]], color=COLOR_CURSOR, s=40, zorder=5,
                       edgecolors="white", linewidths=1)

    # Error ribbon
    error = np.abs(pred - target)
    ax_val.fill_between(time_sec, pred - error * 0.5, pred + error * 0.5,
                        alpha=0.15, color=COLOR_LINE)

    mae = np.abs(pred - target).mean()
    corr = np.corrcoef(pred, target)[0, 1] if T > 1 else 0
    ax_val.text(
        0.02, 0.95, f"MAE={mae:.4f}  Corr={corr:.3f}",
        transform=ax_val.transAxes, fontsize=10, color=COLOR_TEXT,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=COLOR_PANEL, edgecolor="#555", alpha=0.9),
    )

    ax_val.set_xlim(-0.5, time_sec[-1] + 0.5)
    ax_val.set_ylim(-1.05, 0.05)
    ax_val.set_xlabel("Time (s)", fontsize=10, color=COLOR_TEXT)
    ax_val.set_ylabel("Value", fontsize=10, color=COLOR_TEXT)
    ax_val.legend(loc="lower right", fontsize=9, facecolor=COLOR_PANEL,
                  edgecolor="#555", labelcolor=COLOR_TEXT)
    ax_val.tick_params(colors=COLOR_TEXT, labelsize=9)
    ax_val.spines["top"].set_visible(False)
    ax_val.spines["right"].set_visible(False)
    ax_val.spines["bottom"].set_color("#555")
    ax_val.spines["left"].set_color("#555")
    ax_val.grid(True, alpha=0.15, color="#555")

    fig.savefig(output_path, dpi=dpi, facecolor=COLOR_BG)
    plt.close(fig)
    logger.info(f"Static summary saved → {output_path}")


def save_gif(ep_data: dict, output_path: str, fps: int = 10, skip_frames: int = 3):
    """Save an animated GIF of the episode with value curve."""
    T = ep_data["length"]
    frame_indices = list(range(0, T, skip_frames))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)

    logger.info(f"Rendering {len(frame_indices)} gif frames (skip={skip_frames})...")
    pil_frames = []

    for count, fi in enumerate(frame_indices):
        buf = render_frame(ep_data, fi)
        pil_frames.append(Image.fromarray(buf))
        if (count + 1) % 50 == 0:
            logger.info(f"  rendered {count+1}/{len(frame_indices)}")

    duration_ms = int(1000 / fps)
    # Hold last frame longer
    durations = [duration_ms] * len(pil_frames)
    durations[-1] = duration_ms * 5

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    logger.info(f"GIF saved → {output_path} ({len(pil_frames)} frames, {fps} fps)")


def save_mp4(ep_data: dict, output_path: str, fps: int = 10, skip_frames: int = 3):
    """Save an MP4 video of the episode with value curve."""
    import cv2

    T = ep_data["length"]
    frame_indices = list(range(0, T, skip_frames))
    if frame_indices[-1] != T - 1:
        frame_indices.append(T - 1)

    # Render first frame to get dimensions
    first_buf = render_frame(ep_data, frame_indices[0])
    h, w = first_buf.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    logger.info(f"Rendering {len(frame_indices)} video frames...")
    for count, fi in enumerate(frame_indices):
        if count == 0:
            buf = first_buf
        else:
            buf = render_frame(ep_data, fi)
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))
        if (count + 1) % 50 == 0:
            logger.info(f"  rendered {count+1}/{len(frame_indices)}")

    # Hold last frame
    for _ in range(fps * 2):
        writer.write(cv2.cvtColor(buf, cv2.COLOR_RGB2BGR))

    writer.release()
    logger.info(f"MP4 saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize value function on episodes")
    parser.add_argument("--value_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Path to a single LeRobot dataset directory")
    parser.add_argument("--episode_index", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, default="vis_output")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "both"])
    parser.add_argument("--save_png", action="store_true", help="Also save static PNG summary")
    parser.add_argument("--fps", type=int, default=10, help="Output video/gif fps")
    parser.add_argument("--skip_frames", type=int, default=3,
                        help="Process every Nth frame (speeds up rendering)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video_backend", type=str, default="pyav")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    model, config, train_step = load_value_function(args.value_checkpoint, str(device))
    logger.info(f"Loaded value function (step {train_step})")

    # ── Load tokenizer ────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──────────────────────────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    repo_id = Path(args.dataset_root).name
    dataset = LeRobotDataset(repo_id, root=args.dataset_root, video_backend=args.video_backend)

    sample = dataset[0]
    image_keys = [k for k in sample.keys() if "image" in k and isinstance(sample[k], torch.Tensor)]
    logger.info(f"Dataset: {repo_id}, episodes={dataset.meta.info['total_episodes']}, images={image_keys}")

    # ── Process episodes ──────────────────────────────────────────────
    for ep_idx in args.episode_index:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing episode {ep_idx}...")

        ep_data = collect_episode_data(model, dataset, ep_idx, image_keys, tokenizer, device)

        # Summary stats
        pred = ep_data["pred_values"]
        target = ep_data["target_values"]
        mae = np.abs(pred - target).mean()
        corr = np.corrcoef(pred, target)[0, 1] if len(pred) > 1 else 0
        logger.info(f"  MAE={mae:.4f}, Corr={corr:.3f}")
        logger.info(f"  Pred range: [{pred.min():.4f}, {pred.max():.4f}]")
        logger.info(f"  Target range: [{target.min():.4f}, {target.max():.4f}]")

        base_name = f"ep{ep_idx:03d}_{repo_id[:40]}_step{train_step}"

        if args.save_png or True:  # Always save static summary
            png_path = os.path.join(args.output_dir, f"{base_name}_summary.png")
            save_static_summary(ep_data, png_path)

        if args.format in ("gif", "both"):
            gif_path = os.path.join(args.output_dir, f"{base_name}.gif")
            save_gif(ep_data, gif_path, fps=args.fps, skip_frames=args.skip_frames)

        if args.format in ("mp4", "both"):
            mp4_path = os.path.join(args.output_dir, f"{base_name}.mp4")
            save_mp4(ep_data, mp4_path, fps=args.fps, skip_frames=args.skip_frames)

    logger.info(f"\nAll done! Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
