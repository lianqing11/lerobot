#!/usr/bin/env python
"""
Evaluate a trained value function on a dataset.

Reports per-task metrics: cross-entropy loss, MAE, value distribution stats,
and bin-level accuracy. Optionally saves per-episode metrics for further analysis.

Usage:
    python eval_value_function.py \
        --value_checkpoint checkpoints/value_function/checkpoint_005000/value_function.pt \
        --dataset_list_file trainset_config/20260321_twotask_collect_more.txt \
        --output_file eval_results.json
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate RECAP value function")
    parser.add_argument("--value_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_list_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video_backend", type=str, default="pyav")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for quick eval")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────
    model, config, train_step = load_value_function(args.value_checkpoint, str(device))
    logger.info(f"Loaded value function from step {train_step}, variant={config['vlm_variant']}")

    # ── Load tokenizer ────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──────────────────────────────────────────────────
    from train_value_function import ValueFunctionDataset, collate_fn, parse_dataset_list_file

    dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    vf_dataset = ValueFunctionDataset(
        dataset_entries, all_success=True, video_backend=args.video_backend,
    )

    sample = vf_dataset[0]
    image_keys = [k for k in sample.keys() if "image" in k and isinstance(sample[k], torch.Tensor)]

    dataloader = DataLoader(
        vf_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ── Inference ─────────────────────────────────────────────────────
    logger.info(f"Evaluating on {len(vf_dataset)} frames...")
    all_preds = []
    all_targets = []
    all_losses = []
    all_tasks = []
    all_episode_ids = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, img_masks = model.preprocess_images(batch, image_keys)
            tasks = batch.get("task", [""] * args.batch_size)
            encoded = tokenizer(
                tasks, padding="max_length", max_length=200, truncation=True, return_tensors="pt",
            )
            tokens = encoded["input_ids"].to(device)
            masks = encoded["attention_mask"].to(device).bool()
            target_bins = batch["value_target_bin"].long().to(device)

            logits = model.forward(images, img_masks, tokens, masks)
            per_sample_loss = F.cross_entropy(logits, target_bins, reduction="none")

            probs = F.softmax(logits, dim=-1)
            pred_values = (probs * model.bin_values).sum(dim=-1)
            target_values = model.bin_values[target_bins]

            all_preds.append(pred_values.cpu().numpy())
            all_targets.append(target_values.cpu().numpy())
            all_losses.append(per_sample_loss.cpu().numpy())
            all_tasks.extend(tasks if isinstance(tasks, list) else [tasks] * logits.shape[0])

            ep_ids = batch.get("episode_index", torch.zeros(logits.shape[0]))
            if isinstance(ep_ids, torch.Tensor):
                ep_ids = ep_ids.numpy()
            all_episode_ids.extend(ep_ids)

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  {(batch_idx+1)*args.batch_size}/{len(vf_dataset)} frames")

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_losses = np.concatenate(all_losses)
    elapsed = time.time() - t0
    logger.info(f"Inference done in {elapsed:.1f}s ({len(vf_dataset)/elapsed:.0f} frames/s)")

    # ── Global metrics ────────────────────────────────────────────────
    mae = np.abs(all_preds - all_targets).mean()
    mse = ((all_preds - all_targets) ** 2).mean()
    mean_loss = all_losses.mean()
    correlation = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0

    logger.info("=" * 60)
    logger.info(f"  Total frames:  {len(vf_dataset)}")
    logger.info(f"  CE Loss:       {mean_loss:.4f}")
    logger.info(f"  MAE:           {mae:.4f}")
    logger.info(f"  MSE:           {mse:.6f}")
    logger.info(f"  Correlation:   {correlation:.4f}")
    logger.info(f"  Pred  range:   [{all_preds.min():.4f}, {all_preds.max():.4f}]  mean={all_preds.mean():.4f}")
    logger.info(f"  Target range:  [{all_targets.min():.4f}, {all_targets.max():.4f}]  mean={all_targets.mean():.4f}")

    # ── Per-task metrics ──────────────────────────────────────────────
    task_metrics = {}
    unique_tasks = sorted(set(all_tasks))
    for task in unique_tasks:
        mask = np.array([t == task for t in all_tasks])
        t_preds = all_preds[mask]
        t_targets = all_targets[mask]
        t_losses = all_losses[mask]
        t_mae = np.abs(t_preds - t_targets).mean()
        t_corr = np.corrcoef(t_preds, t_targets)[0, 1] if len(t_preds) > 1 else 0

        task_metrics[task] = {
            "n_frames": int(mask.sum()),
            "ce_loss": float(t_losses.mean()),
            "mae": float(t_mae),
            "correlation": float(t_corr),
            "pred_mean": float(t_preds.mean()),
            "target_mean": float(t_targets.mean()),
        }
        logger.info(f"  Task: {task}")
        logger.info(f"    frames={mask.sum()}, loss={t_losses.mean():.4f}, mae={t_mae:.4f}, corr={t_corr:.4f}")

    logger.info("=" * 60)

    # ── Save ──────────────────────────────────────────────────────────
    results = {
        "checkpoint": args.value_checkpoint,
        "train_step": train_step,
        "global": {
            "n_frames": len(vf_dataset),
            "ce_loss": float(mean_loss),
            "mae": float(mae),
            "mse": float(mse),
            "correlation": float(correlation),
        },
        "per_task": task_metrics,
    }

    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(args.value_checkpoint).parent / "eval_results.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {output_file}")


if __name__ == "__main__":
    main()
