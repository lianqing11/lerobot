#!/usr/bin/env python
"""
Evaluate a trained value function on a dataset.

Reports per-task metrics: cross-entropy loss, MAE, value distribution stats,
and bin-level accuracy. Optionally saves per-episode metrics for further analysis.

Usage:
    python -m lerobot.value_function.eval_value_function \
        --value_checkpoint checkpoints/value_function/checkpoint_005000/value_function.pt \
        --dataset_list_file trainset_config/20260321_twotask_collect_more.txt \
        --output_file eval_results.json
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader, Subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        if (checkpoint / "value_function.pt").exists():
            return checkpoint / "value_function.pt"
        if (checkpoint / "training_state.pt").exists():
            return checkpoint / "training_state.pt"
    return checkpoint


def load_value_function(
    checkpoint_path: str,
    device: str = "cuda",
    precision: str | None = None,
    attn_implementation: str | None = None,
):
    from lerobot.value_function.value_function_pi05 import PI05ValueFunction

    resolved = _resolve_checkpoint_path(checkpoint_path)
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = PI05ValueFunction(
        backbone_family=config.get("backbone_family", "paligemma"),
        pretrained_model_name=config.get("pretrained_model_name", "Qwen/Qwen3-VL-2B-Instruct"),
        vlm_variant=config.get("vlm_variant", "gemma_300m"),
        image_resolution=(224, 224),
        precision=precision or config.get("precision", "float32"),
        load_pretrained_backbone=False,
        attn_implementation=attn_implementation or config.get("attn_implementation"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Extract training-time normalization constants (saved by newer checkpoints).
    train_norm = {
        "max_length_per_task": ckpt.get("max_length_per_task"),
    }
    return model, config, ckpt["step"], train_norm


def main():
    parser = argparse.ArgumentParser(description="Evaluate RECAP value function")
    parser.add_argument("--value_checkpoint", type=str, required=True)
    parser.add_argument("--dataset_list_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager", "default"],
    )
    parser.add_argument("--video_backend", type=str, default="pyav")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for quick eval")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load model ────────────────────────────────────────────────────
    model, config, train_step, train_norm = load_value_function(
        args.value_checkpoint,
        str(device),
        precision=args.precision,
        attn_implementation=args.attn_implementation,
    )
    logger.info(
        "Loaded value function from step %s, backbone=%s",
        train_step,
        config.get("backbone_family", "paligemma"),
    )
    if train_norm["max_length_per_task"]:
        logger.info("Using training max_length_per_task: %s", train_norm["max_length_per_task"])
    else:
        logger.warning(
            "Checkpoint has no max_length_per_task — value targets will be normalized "
            "by test-set episode lengths, which may differ from training."
        )

    # ── Load tokenizer ────────────────────────────────────────────────
    # ── Load dataset ──────────────────────────────────────────────────
    from lerobot.value_function.train_value_function import (
        ValueFunctionDataset,
        collate_fn,
        frame_indices_from_episode_infos,
        load_value_processor,
        prepare_value_inputs,
        parse_dataset_list_file,
    )

    processor = load_value_processor(
        config.get("backbone_family", "paligemma"),
        config.get("pretrained_model_name", "Qwen/Qwen3-VL-2B-Instruct"),
    )

    dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    base_dataset = ValueFunctionDataset(
        dataset_entries, all_success=True, video_backend=args.video_backend,
        max_length_per_task_override=train_norm["max_length_per_task"],
    )
    vf_dataset = base_dataset

    if args.max_episodes is not None:
        limited_episode_infos = base_dataset.episode_infos[:args.max_episodes]
        limited_frame_indices = frame_indices_from_episode_infos(limited_episode_infos)
        vf_dataset = Subset(base_dataset, limited_frame_indices)
        logger.info(
            f"Quick eval enabled: {len(limited_episode_infos)} episodes, {len(limited_frame_indices)} frames"
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
            prepared_inputs = prepare_value_inputs(
                batch, processor, model, image_keys, device=str(device)
            )
            target_bins = batch["value_target_bin"].long().to(device)

            logits = model.forward_from_prepared_inputs(prepared_inputs)
            per_sample_loss = F.cross_entropy(logits, target_bins, reduction="none")

            probs = F.softmax(logits, dim=-1)
            pred_values = (probs * model.bin_values).sum(dim=-1)
            target_values = model.bin_values[target_bins]

            all_preds.append(pred_values.cpu().numpy())
            all_targets.append(target_values.cpu().numpy())
            all_losses.append(per_sample_loss.cpu().numpy())
            tasks = batch.get("task", ["unknown"] * logits.shape[0])
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
