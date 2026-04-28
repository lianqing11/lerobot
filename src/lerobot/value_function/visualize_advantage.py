#!/usr/bin/env python
"""Visualize KAI-0-style advantage function predictions.

For each episode, computes:
  1. Absolute value: model(frame_0, frame_n) — total progress from start
  2. Relative advantage: model(frame_n, frame_{n+interval}) — local progress rate

Output: visualization/advantage_head/{model_name}_{step}/{dataset_name}_ep{NNN}.mp4
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

for proxy_key in ("ALL_PROXY", "all_proxy"):
    os.environ.pop(proxy_key, None)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.value_function.train_advantage_function import (
    _extract_episode_task_name,
    collate_fn,
    format_advantage_prompts,
    load_advantage_processor,
    parse_dataset_list_file,
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

# ── Checkpoint loading ──────────────────────────────────────────────


def load_advantage_function(checkpoint_path, device="cuda", precision=None, attn_implementation=None):
    from lerobot.value_function.advantage_function_pi05 import PI05AdvantageFunction

    p = Path(checkpoint_path)
    if p.is_dir():
        p = p / "advantage_function.pt" if (p / "advantage_function.pt").exists() else p / "training_state.pt"
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = PI05AdvantageFunction(
        backbone_family=config.get("backbone_family", "qwen_vl"),
        pretrained_model_name=config.get("pretrained_model_name", "Qwen/Qwen3-VL-2B-Instruct"),
        vlm_variant=config.get("vlm_variant", "gemma_300m"),
        image_resolution=(224, 224),
        precision=precision or config.get("precision", "float32"),
        load_pretrained_backbone=False,
        attn_implementation=attn_implementation or config.get("attn_implementation"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, config, ckpt["step"]


# ── Dataset ─────────────────────────────────────────────────────────


class AdvantageVisDataset(Dataset):
    def __init__(self, dataset_entries, video_backend="pyav"):
        self.sub_datasets = []
        self.episode_infos: list[EpisodeInfo] = []

        for repo_id, dataset_root in dataset_entries:
            ds = LeRobotDataset(repo_id, root=str(Path(dataset_root).resolve()), revision="main", video_backend=video_backend)
            ds_idx = len(self.sub_datasets)
            self.sub_datasets.append(ds)
            for ep in ds.meta.episodes:
                ep_idx, length = int(ep["episode_index"]), int(ep["length"])
                task_name = _extract_episode_task_name(ds, ep)
                self.episode_infos.append(EpisodeInfo(
                    ds_idx=ds_idx, repo_id=repo_id, episode_index=ep_idx,
                    task_name=task_name, length=length,
                    dataset_from_index=int(ep["dataset_from_index"]),
                    dataset_to_index=int(ep["dataset_to_index"]),
                ))


# ── Inference ───────────────────────────────────────────────────────


def _build_paired_batch(dataset, episode_info, cur_indices, ref_indices, image_keys):
    ds = dataset.sub_datasets[episode_info.ds_idx]
    items = []
    for cur_local, ref_local in zip(cur_indices, ref_indices, strict=True):
        cur_item = ds[episode_info.dataset_from_index + cur_local]
        ref_item = ds[episode_info.dataset_from_index + ref_local]
        item = dict(cur_item)
        for key in image_keys:
            if key in ref_item:
                item[f"ref_{key}"] = ref_item[key]
        items.append(item)
    return collate_fn(items)


def infer_episode_advantages(model, processor, dataset, ep, image_keys, batch_size, device, relative_interval=50):
    n = ep.length
    gt_progress = np.array([i / max(n - 1, 1) for i in range(n)])
    abs_values = np.zeros(n)
    rel_advantages = np.zeros(n)
    ref_image_keys = [f"ref_{k}" for k in image_keys]

    with torch.inference_mode():
        # Absolute: model(frame_0, frame_n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cur = list(range(start, end))
            ref = [0] * len(cur)
            batch = _build_paired_batch(dataset, ep, cur, ref, image_keys)
            prompts = format_advantage_prompts([ep.task_name] * len(cur))
            prepared = model.prepare_inputs(batch, image_keys, ref_image_keys, processor, prompts, device=device)
            pred = model.predict_advantage(prepared).cpu().numpy()
            for i, idx in enumerate(cur):
                abs_values[idx] = 0.0 if idx == 0 else float(pred[i])

        # Relative: model(frame_n, frame_{n+interval})
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            cur = list(range(start, end))
            future = [min(i + relative_interval, n - 1) for i in cur]
            batch = _build_paired_batch(dataset, ep, future, cur, image_keys)
            prompts = format_advantage_prompts([ep.task_name] * len(cur))
            prepared = model.prepare_inputs(batch, image_keys, ref_image_keys, processor, prompts, device=device)
            pred = model.predict_advantage(prepared).cpu().numpy()
            for i, idx in enumerate(cur):
                gap = future[i] - idx
                if gap == 0:
                    rel_advantages[idx] = 0.0
                elif gap != relative_interval:
                    rel_advantages[idx] = np.clip(float(pred[i]) / gap * relative_interval, -1.0, 1.0)
                else:
                    rel_advantages[idx] = np.clip(float(pred[i]), -1.0, 1.0)

    return gt_progress, abs_values, rel_advantages


# ── Rendering ───────────────────────────────────────────────────────


def render_frame_panel(main_image, wrist_image, gt, abs_v, rel_v, current_index, episode_label, metrics_text):
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
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = int(round(pt + ph - ((v + 1.0) / 2.0 * ph)))
        draw.line((pl, y, pr, y), fill=(180, 180, 180, 92), width=1)
        text_draw.text((pl + 6, max(top_bar_h + 6, y - 8)), f"{v:.1f}", fill=(245, 245, 245), font=font)

    abs_c = (0, 210, 255, 255)
    pts = xy_points(abs_v, pl, pt, pw, ph, vmin=-1.0, vmax=1.0)
    if len(pts) >= 2:
        draw.line(pts, fill=abs_c, width=4)

    x_cur = pl + pw // 2 if len(abs_v) == 1 else int(round(pl + pw * current_index / (len(abs_v) - 1)))
    draw.line((x_cur, pt, x_cur, pb), fill=(255, 255, 255, 165), width=2)
    y = int(round(pt + ph - ((abs_v[current_index] + 1.0) / 2.0 * ph)))
    draw.ellipse((x_cur - 5, y - 5, x_cur + 5, y + 5), fill=abs_c)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    td = ImageDraw.Draw(canvas)
    td.rectangle((0, 0, 1280, top_bar_h), fill=(8, 8, 8))
    td.line((10, 10, 32, 10), fill=abs_c[:3], width=4)
    td.text((38, 4), "Prediction", fill=(242, 242, 242), font=font)
    td.text((520, 4), "main camera", fill=(242, 242, 242), font=font)
    td.text((1015, 4), "wrist camera", fill=(242, 242, 242), font=font)
    td.text((12, 26), episode_label, fill=(255, 255, 255), font=font)
    td.text((12, 40), metrics_text, fill=(255, 255, 255), font=font)
    return np.asarray(canvas, dtype=np.uint8)


def render_frame_overlay(main_image, wrist_image, gt, abs_v, rel_v, current_index):
    main = tensor_image_to_uint8(main_image)
    wrist = tensor_image_to_uint8(wrist_image)
    merged = np.concatenate([main, wrist], axis=1)

    top_bar_h = 16
    canvas = Image.new("RGB", (merged.shape[1], merged.shape[0] + top_bar_h), (16, 34, 52))
    canvas.paste(Image.fromarray(merged), (0, top_bar_h))
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    shadow = (0, 0, 0, 180)

    pl, pw = 0, merged.shape[1] - 1
    pt = top_bar_h + int(round(merged.shape[0] * 0.25))
    ph = int(round(merged.shape[0] * 0.43))

    abs_c = (0, 210, 255, 255)
    pts = history_points(abs_v, current_index, pl, pt, pw, ph, vmin=-1.0, vmax=1.0)
    if len(pts) >= 2:
        draw.line(pts, fill=shadow, width=4)
        draw.line(pts, fill=abs_c, width=2)
    if pts:
        x, y = pts[-1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=shadow)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=abs_c)

    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    td = ImageDraw.Draw(canvas)
    td.rectangle((0, 0, canvas.size[0], top_bar_h), fill=(19, 40, 62))
    td.text((10, 2), "Prediction", fill=(242, 242, 242), font=font)
    td.ellipse((2, 4, 8, 10), fill=abs_c[:3])
    draw_camera_label(td, 10, top_bar_h + 6, "Main Camera", font)
    draw_camera_label(td, merged.shape[1] // 2 + 10, top_bar_h + 6, "Wrist Camera", font)
    return np.asarray(canvas, dtype=np.uint8)


def render_frame(main_image, wrist_image, gt, abs_v, rel_v, current_index, episode_label, metrics_text, style):
    if style == "overlay":
        return render_frame_overlay(main_image, wrist_image, gt, abs_v, rel_v, current_index)
    return render_frame_panel(main_image, wrist_image, gt, abs_v, rel_v, current_index, episode_label, metrics_text)


def render_episode_video(output_path, dataset, ep, gt, abs_v, rel_v, fps, skip_frames, metrics_text, style):
    frames = []
    ds = dataset.sub_datasets[ep.ds_idx]
    label = f"{ep.repo_id} | ep {ep.episode_index:03d} | {ep.task_name} | len {ep.length}"
    for i in range(0, ep.length, skip_frames):
        item = ds[ep.dataset_from_index + i]
        frames.append(render_frame(item["observation.images.main"], item["observation.images.wrist"],
                                   gt, abs_v, rel_v, i, label, metrics_text, style))
    if ep.length > 0 and (ep.length - 1) % skip_frames != 0:
        item = ds[ep.dataset_from_index + ep.length - 1]
        frames.append(render_frame(item["observation.images.main"], item["observation.images.wrist"],
                                   gt, abs_v, rel_v, ep.length - 1, label, metrics_text, style))
    save_video(frames, output_path, fps)


# ── Main ────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Visualize advantage function predictions")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None, help="If omitted, auto-derived from checkpoint")
    p.add_argument("--dataset_roots", nargs="+", default=None)
    p.add_argument("--dataset_list_file", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--attn_implementation", type=str, default="auto",
                    choices=["auto", "flash_attention_2", "sdpa", "eager", "default"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--video_backend", type=str, default="pyav")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--skip_frames", type=int, default=3)
    p.add_argument("--relative_interval", type=int, default=50)
    p.add_argument("--selection_mode", type=str, default="all",
                    choices=["all", "per_task_first", "per_task_longest", "per_task_topk"])
    p.add_argument("--episodes_per_group", type=int, default=1)
    p.add_argument("--progress_style", type=str, default="panel", choices=["overlay", "panel"])
    p.add_argument("--filename_prefix", type=str, default="", help="Prefix for output video filenames")
    args = p.parse_args()

    if args.dataset_list_file:
        dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    elif args.dataset_roots:
        dataset_entries = [(Path(p_).resolve().name, str(Path(p_).resolve())) for p_ in args.dataset_roots]
    else:
        raise ValueError("Provide either --dataset_list_file or --dataset_roots")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else derive_output_dir(args.checkpoint, "advantage_head")
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config, train_step = load_advantage_function(
        args.checkpoint, device=args.device, precision=args.precision,
        attn_implementation=args.attn_implementation,
    )
    processor = load_advantage_processor(config.get("backbone_family", "qwen_vl"),
                                          config.get("pretrained_model_name", "Qwen/Qwen3-VL-2B-Instruct"))
    dataset = AdvantageVisDataset(dataset_entries, video_backend=args.video_backend)
    sample = dataset.sub_datasets[0][0]
    image_keys = config.get("image_keys", [k for k, v in sample.items() if "image" in k and isinstance(v, torch.Tensor)])
    selected = select_episode_infos(dataset.episode_infos, args.selection_mode, args.episodes_per_group)

    summary = {"checkpoint": str(Path(args.checkpoint).resolve()), "train_step": int(train_step),
               "config": config, "selected_episode_count": len(selected), "global": {}, "per_episode": []}

    for idx, ep in enumerate(selected, start=1):
        gt, abs_v, rel_v = infer_episode_advantages(
            model, processor, dataset, ep, image_keys, args.batch_size, args.device, args.relative_interval)
        abs_mae = float(np.mean(np.abs(abs_v - gt)))
        abs_corr = safe_corrcoef(abs_v, gt)
        metrics_text = f"Abs MAE {abs_mae:.4f} | Abs Corr {abs_corr:.3f}"
        vpath = output_dir / video_filename(ep, prefix=args.filename_prefix)
        render_episode_video(vpath, dataset, ep, gt, abs_v, rel_v, args.fps, args.skip_frames, metrics_text, args.progress_style)
        summary["per_episode"].append({
            "repo_id": ep.repo_id, "episode_index": ep.episode_index, "task_name": ep.task_name,
            "length": ep.length, "abs_mae": abs_mae, "abs_corr": abs_corr,
            "rel_mean": float(np.mean(rel_v)), "video_path": str(vpath),
        })
        print(f"[{idx}/{len(selected)}] {ep.repo_id} ep{ep.episode_index:03d} abs_mae={abs_mae:.4f} abs_corr={abs_corr:.3f} -> {vpath.name}", flush=True)

    all_mae = [e["abs_mae"] for e in summary["per_episode"]]
    all_corr = [e["abs_corr"] for e in summary["per_episode"]]
    summary["global"] = {
        "episodes": len(summary["per_episode"]),
        "abs_mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "abs_corr": float(np.mean(all_corr)) if all_corr else 0.0,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary["global"], indent=2))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
