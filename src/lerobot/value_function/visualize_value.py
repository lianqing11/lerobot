#!/usr/bin/env python
"""Visualize value head predictions. Renders per-episode MP4s with prediction overlay.

Usage:
    # Auto-derive output dir from checkpoint
    python -m lerobot.value_function.visualize_value \
        --checkpoint ckpt/value_head_baseline_qwen3vl_20260413_150840/checkpoint_015000 \
        --dataset_roots /path/to/dataset1 /path/to/dataset2

    # Explicit output dir
    python -m lerobot.value_function.visualize_value \
        --checkpoint ... --dataset_list_file datasets.txt --output_dir my_vis/

Output: visualization/value_head/{model_name}_{step}/{dataset_name}_ep{NNN}.mp4
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Subset

for proxy_key in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(proxy_key, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.value_function.eval_value_function import load_value_function
from lerobot.value_function.train_value_function import (
    _extract_episode_task_name,
    collate_fn,
    load_value_processor,
    parse_dataset_list_file,
    prepare_value_inputs,
)
from lerobot.value_function.vis_utils import (
    EpisodeInfo,
    derive_output_dir,
    draw_camera_label,
    history_points,
    safe_corrcoef,
    save_video,
    select_episode_infos,
    tensor_image_to_uint8,
    video_filename,
    xy_points,
)

# ── Dataset ─────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class FrameInfo:
    ds_idx: int
    repo_id: str
    episode_index: int
    task_name: str
    frame_in_episode: int
    global_frame_idx: int
    value_target_bin: int
    value_target: float


class ValueVisDataset(Dataset):
    def __init__(self, dataset_entries, num_bins=201, video_backend="pyav", max_length_per_task_override=None):
        self.num_bins = num_bins
        self.sub_datasets = []
        self.episode_infos: list[EpisodeInfo] = []
        self.frame_map: list[FrameInfo] = []
        max_length_per_task: dict[str, int] = {}

        for repo_id, dataset_root in dataset_entries:
            ds = LeRobotDataset(repo_id, root=str(Path(dataset_root).resolve()), revision="main", video_backend=video_backend)
            ds_idx = len(self.sub_datasets)
            self.sub_datasets.append(ds)
            for ep in ds.meta.episodes:
                ep_idx, length = int(ep["episode_index"]), int(ep["length"])
                task_name = _extract_episode_task_name(ds, ep)
                max_length_per_task[task_name] = max(max_length_per_task.get(task_name, 0), length)
                self.episode_infos.append(EpisodeInfo(
                    ds_idx=ds_idx, repo_id=repo_id, episode_index=ep_idx,
                    task_name=task_name, length=length,
                    dataset_from_index=int(ep["dataset_from_index"]),
                    dataset_to_index=int(ep["dataset_to_index"]),
                ))

        if max_length_per_task_override is not None:
            for task in max_length_per_task:
                if task in max_length_per_task_override:
                    max_length_per_task[task] = max_length_per_task_override[task]
            for task, val in max_length_per_task_override.items():
                if task not in max_length_per_task:
                    max_length_per_task[task] = val
        self.max_length_per_task = max_length_per_task

        for info in self.episode_infos:
            max_len = max_length_per_task.get(info.task_name, info.length)
            info.max_length = max_len
            info.frame_start = len(self.frame_map)
            for frame_in_ep in range(info.length):
                steps_remaining = info.length - 1 - frame_in_ep
                normalized_return = max(-1.0, min(0.0, -steps_remaining / max_len))
                bin_idx = max(0, min(num_bins - 1, int((normalized_return + 1.0) * (num_bins - 1))))
                self.frame_map.append(FrameInfo(
                    ds_idx=info.ds_idx, repo_id=info.repo_id, episode_index=info.episode_index,
                    task_name=info.task_name, frame_in_episode=frame_in_ep,
                    global_frame_idx=info.dataset_from_index + frame_in_ep,
                    value_target_bin=bin_idx, value_target=float(normalized_return),
                ))
            info.frame_end = len(self.frame_map)

    def __len__(self):
        return len(self.frame_map)

    def __getitem__(self, idx):
        fi = self.frame_map[idx]
        item = self.sub_datasets[fi.ds_idx][fi.global_frame_idx]
        item["value_target_bin"] = fi.value_target_bin
        item["value_target"] = fi.value_target
        return item


# ── Inference ───────────────────────────────────────────────────────


def infer_episode(model, processor, dataset, episode_info, image_keys, batch_size, num_workers, device):
    indices = list(range(episode_info.frame_start, episode_info.frame_end))
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)
    preds, targets, losses = [], [], []
    with torch.inference_mode():
        for batch in loader:
            prepared = prepare_value_inputs(batch, processor, model, image_keys, device=device)
            target_bins = batch["value_target_bin"].long().to(device)
            logits = model.forward_from_prepared_inputs(prepared)
            probs = F.softmax(logits, dim=-1)
            preds.append((probs * model.bin_values).sum(dim=-1).cpu().numpy())
            targets.append(model.bin_values[target_bins].cpu().numpy())
            losses.append(F.cross_entropy(logits, target_bins, reduction="none").cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets), np.concatenate(losses)


# ── Rendering ───────────────────────────────────────────────────────


def render_frame_panel(main_image, wrist_image, pred_values, current_index, episode_label, metrics_text):
    main = tensor_image_to_uint8(main_image)
    wrist = tensor_image_to_uint8(wrist_image)
    merged = np.concatenate([main, wrist], axis=1)

    top_bar_h = 20
    canvas = Image.new("RGB", (1280, 496), (0, 0, 0))
    canvas.paste(Image.fromarray(merged), (0, top_bar_h))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    text_draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    pl, pr, pt, pb = 18, 1262, 292, 490
    pw, ph = pr - pl, pb - pt
    draw.rectangle((pl, pt, pr, pb), fill=(0, 0, 0, 140))
    for v in (0.0, -0.5, -1.0):
        y = int(round(pt + (0.0 - v) * ph))
        draw.line((pl, y, pr, y), fill=(180, 180, 180, 92), width=1)
        text_draw.text((pl + 6, max(top_bar_h + 6, y - 8)), f"{v:.1f}", fill=(245, 245, 245), font=font)

    pred_color = (0, 210, 255, 255)
    pts = xy_points(pred_values, pl, pt, pw, ph)
    if len(pts) >= 2:
        draw.line(pts, fill=pred_color, width=4)
    x_cur = pl + pw // 2 if len(pred_values) == 1 else int(round(pl + pw * current_index / (len(pred_values) - 1)))
    draw.line((x_cur, pt, x_cur, pb), fill=(255, 255, 255, 165), width=2)
    y_pred = int(round(pt + (0.0 - pred_values[current_index]) * ph))
    draw.ellipse((x_cur - 5, y_pred - 5, x_cur + 5, y_pred + 5), fill=pred_color)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    td = ImageDraw.Draw(canvas)
    td.rectangle((0, 0, 1280, top_bar_h), fill=(8, 8, 8))
    td.line((10, 10, 32, 10), fill=pred_color[:3], width=4)
    td.text((38, 4), "Prediction", fill=(242, 242, 242), font=font)
    td.text((520, 4), "main camera", fill=(242, 242, 242), font=font)
    td.text((1015, 4), "wrist camera", fill=(242, 242, 242), font=font)
    td.text((12, 26), episode_label, fill=(255, 255, 255), font=font)
    td.text((12, 40), metrics_text, fill=(255, 255, 255), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def render_frame_overlay(main_image, wrist_image, pred_values, current_index):
    main = tensor_image_to_uint8(main_image)
    wrist = tensor_image_to_uint8(wrist_image)
    merged = np.concatenate([main, wrist], axis=1)

    top_bar_h = 16
    canvas = Image.new("RGB", (merged.shape[1], merged.shape[0] + top_bar_h), (16, 34, 52))
    canvas.paste(Image.fromarray(merged), (0, top_bar_h))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    pred_color = (255, 212, 46, 255)
    shadow = (0, 0, 0, 180)
    pl, pw = 0, merged.shape[1] - 1
    pt = top_bar_h + int(round(merged.shape[0] * 0.25))
    ph = int(round(merged.shape[0] * 0.43))

    pts = history_points(pred_values, current_index, pl, pt, pw, ph)
    if len(pts) >= 2:
        draw.line(pts, fill=shadow, width=4)
        draw.line(pts, fill=pred_color, width=2)
    if pts:
        x, y = pts[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=shadow)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=pred_color)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    td = ImageDraw.Draw(canvas)
    td.rectangle((0, 0, canvas.size[0], top_bar_h), fill=(19, 40, 62))
    td.text((10, 2), "Prediction", fill=(242, 242, 242), font=font)
    td.ellipse((2, 4, 8, 10), fill=pred_color[:3])
    draw_camera_label(td, 10, top_bar_h + 6, "Main Camera", font)
    draw_camera_label(td, merged.shape[1] // 2 + 10, top_bar_h + 6, "Wrist Camera", font)
    return np.asarray(canvas, dtype=np.uint8)


def render_frame(main_image, wrist_image, pred_values, current_index, episode_label, metrics_text, style):
    if style == "overlay":
        return render_frame_overlay(main_image, wrist_image, pred_values, current_index)
    return render_frame_panel(main_image, wrist_image, pred_values, current_index, episode_label, metrics_text)


def render_episode_video(output_path, dataset, ep, pred_values, fps, skip_frames, mae, corr, style):
    frames = []
    ds = dataset.sub_datasets[ep.ds_idx]
    label = f"{ep.repo_id} | ep {ep.episode_index:03d} | {ep.task_name} | len {ep.length}"
    metrics = f"MAE {mae:.4f} | Corr {corr:.3f}"
    for i in range(0, ep.length, skip_frames):
        item = ds[dataset.frame_map[ep.frame_start + i].global_frame_idx]
        frames.append(render_frame(item["observation.images.main"], item["observation.images.wrist"],
                                   pred_values, i, label, metrics, style))
    if ep.length > 0 and (ep.length - 1) % skip_frames != 0:
        item = ds[dataset.frame_map[ep.frame_start + ep.length - 1].global_frame_idx]
        frames.append(render_frame(item["observation.images.main"], item["observation.images.wrist"],
                                   pred_values, ep.length - 1, label, metrics, style))
    save_video(frames, output_path, fps)


# ── Main ────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Visualize value head predictions")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None, help="If omitted, auto-derived from checkpoint")
    p.add_argument("--dataset_roots", nargs="+", default=None)
    p.add_argument("--dataset_list_file", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--attn_implementation", type=str, default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager", "default"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--video_backend", type=str, default="pyav")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--skip_frames", type=int, default=3)
    p.add_argument("--selection_mode", type=str, default="all",
                    choices=["all", "per_task_first", "per_task_longest", "per_task_topk"])
    p.add_argument("--episodes_per_group", type=int, default=1)
    p.add_argument("--progress_style", type=str, default="overlay", choices=["overlay", "panel"])
    p.add_argument("--filename_prefix", type=str, default="", help="Prefix for output video filenames (e.g. 'train_data')")
    args = p.parse_args()

    if args.dataset_list_file:
        dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    elif args.dataset_roots:
        dataset_entries = [(Path(p_).resolve().name, str(Path(p_).resolve())) for p_ in args.dataset_roots]
    else:
        raise ValueError("Provide either --dataset_list_file or --dataset_roots")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else derive_output_dir(args.checkpoint, "value_head")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, train_step, train_norm = load_value_function(
        args.checkpoint, device=args.device, precision=args.precision,
        attn_implementation=args.attn_implementation,
    )
    processor = load_value_processor(config.get("backbone_family", "qwen_vl"),
                                     config.get("pretrained_model_name", "Qwen/Qwen3-VL-2B-Instruct"))
    dataset = ValueVisDataset(dataset_entries, num_bins=int(config.get("num_bins", 201)),
                              video_backend=args.video_backend,
                              max_length_per_task_override=train_norm["max_length_per_task"])
    sample = dataset[0]
    image_keys = config.get("image_keys", [k for k, v in sample.items() if "image" in k and isinstance(v, torch.Tensor)])
    selected = select_episode_infos(dataset.episode_infos, args.selection_mode, args.episodes_per_group)

    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "train_step": int(train_step),
               "config": config, "selected_episode_count": len(selected), "global": {}, "per_episode": []}

    for idx, ep in enumerate(selected, start=1):
        pred_ep, target_ep, loss_ep = infer_episode(model, processor, dataset, ep, image_keys,
                                                     args.batch_size, args.num_workers, args.device)
        mae = float(np.mean(np.abs(pred_ep - target_ep)))
        corr = safe_corrcoef(pred_ep, target_ep)
        vpath = output_dir / video_filename(ep, prefix=args.filename_prefix)
        render_episode_video(vpath, dataset, ep, pred_ep, args.fps, args.skip_frames, mae, corr, args.progress_style)
        summary["per_episode"].append({
            "repo_id": ep.repo_id, "episode_index": ep.episode_index, "task_name": ep.task_name,
            "length": ep.length, "n_frames": len(pred_ep), "mae": mae, "correlation": corr,
            "ce_loss": float(np.mean(loss_ep)), "video_path": str(vpath),
        })
        print(f"[{idx}/{len(selected)}] {ep.repo_id} ep{ep.episode_index:03d} mae={mae:.4f} corr={corr:.3f} -> {vpath.name}", flush=True)

    all_mae = [e["mae"] for e in summary["per_episode"]]
    all_corr = [e["correlation"] for e in summary["per_episode"]]
    all_loss = [e["ce_loss"] for e in summary["per_episode"]]
    summary["global"] = {
        "episodes": len(summary["per_episode"]),
        "frames": sum(e["n_frames"] for e in summary["per_episode"]),
        "mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "correlation": float(np.mean(all_corr)) if all_corr else 0.0,
        "ce_loss": float(np.mean(all_loss)) if all_loss else 0.0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["global"], indent=2))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
