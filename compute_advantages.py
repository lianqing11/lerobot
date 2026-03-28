#!/usr/bin/env python
"""
Compute advantage values using a trained value function and save them for
advantage-conditioned policy training (RECAP).

This script:
1. Loads a trained value function checkpoint
2. Runs inference on every frame in the dataset
3. Computes N-step advantage: A(o_t, a_t) = Σ r_{t'} + V(o_{t+N}) - V(o_t)
4. Binarizes advantages with a per-task percentile threshold
5. Saves the advantage annotations as a JSON file

Usage:
    python compute_advantages.py \
        --dataset_list_file trainset_config/20260321_twotask_collect_more.txt \
        --value_checkpoint checkpoints/value_function/checkpoint_005000/value_function.pt \
        --output_file advantages.json \
        --n_lookahead 50 \
        --positive_percentile 30
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compute advantages for RECAP")
    parser.add_argument("--dataset_list_file", type=str, required=True)
    parser.add_argument("--value_checkpoint", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="advantages.json")
    parser.add_argument("--n_lookahead", type=int, default=50,
                        help="N-step lookahead for advantage estimation")
    parser.add_argument("--positive_percentile", type=float, default=30.0,
                        help="Percentile threshold: top (100-p)%% of advantages are 'positive'")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--video_backend", type=str, default="pyav")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── Load value function ───────────────────────────────────────────
    from lerobot.policies.pi05.value_function_pi05 import PI05ValueFunction

    ckpt = torch.load(args.value_checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = PI05ValueFunction(
        vlm_variant=config["vlm_variant"],
        image_resolution=(224, 224),
        precision=config.get("precision", "float32"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info(f"Loaded value function from {args.value_checkpoint} (step {ckpt['step']})")

    # ── Load tokenizer ────────────────────────────────────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load dataset ──────────────────────────────────────────────────
    from train_value_function import ValueFunctionDataset, collate_fn, parse_dataset_list_file

    dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    vf_dataset = ValueFunctionDataset(
        dataset_entries,
        all_success=True,
        video_backend=args.video_backend,
    )

    sample = vf_dataset[0]
    image_keys = [k for k in sample.keys() if "image" in k and isinstance(sample[k], torch.Tensor)]
    logger.info(f"Image features: {image_keys}")

    dataloader = DataLoader(
        vf_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Run value function inference on all frames ────────────────────
    logger.info("Running value function inference on all frames...")
    all_values = []
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, img_masks = model.preprocess_images(batch, image_keys)

            tasks = batch.get("task", [""] * args.batch_size)
            encoded = tokenizer(
                tasks,
                padding="max_length",
                max_length=200,
                truncation=True,
                return_tensors="pt",
            )
            tokens = encoded["input_ids"].to(device)
            masks = encoded["attention_mask"].to(device).bool()

            values = model.predict_value(images, img_masks, tokens, masks)
            all_values.append(values.cpu())

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Processed {(batch_idx+1) * args.batch_size} / {len(vf_dataset)} frames")

    all_values = torch.cat(all_values, dim=0).numpy()
    logger.info(f"Inference done in {time.time()-t0:.1f}s. Values shape: {all_values.shape}")
    logger.info(f"Value stats: mean={all_values.mean():.4f}, std={all_values.std():.4f}, "
                f"min={all_values.min():.4f}, max={all_values.max():.4f}")

    # ── Compute N-step advantages per episode ─────────────────────────
    # Build episode boundaries from frame_map
    episode_frames = defaultdict(list)  # (ds_idx, ep_idx) → list of (frame_pos_in_episode, global_frame_idx, value_idx)
    for value_idx, (ds_idx, global_frame_idx, bin_idx, return_val) in enumerate(vf_dataset.frame_map):
        # We need to figure out which episode this frame belongs to
        ds = vf_dataset.sub_datasets[ds_idx]
        ep_data = None
        for ep in ds.meta.episodes:
            if ep["dataset_from_index"] <= global_frame_idx < ep["dataset_to_index"]:
                ep_data = ep
                break
        if ep_data is None:
            continue
        ep_idx = ep_data["episode_index"]
        frame_in_ep = global_frame_idx - ep_data["dataset_from_index"]
        episode_frames[(ds_idx, ep_idx)].append((frame_in_ep, value_idx, return_val))

    logger.info(f"Computing {args.n_lookahead}-step advantages for {len(episode_frames)} episodes...")

    N = args.n_lookahead
    all_advantages = np.zeros(len(vf_dataset), dtype=np.float32)
    task_advantages = defaultdict(list)  # task_name → list of (value_idx, advantage)

    for (ds_idx, ep_idx), frames in episode_frames.items():
        frames.sort(key=lambda x: x[0])  # sort by frame_in_ep
        ep_length = len(frames)

        # Get task name
        ds = vf_dataset.sub_datasets[ds_idx]
        task_name = str(ds.meta.tasks.index[0]) if ds.meta.tasks is not None and len(ds.meta.tasks) > 0 else "unknown"

        for i, (frame_in_ep, value_idx, return_val) in enumerate(frames):
            v_t = all_values[value_idx]

            # N-step return: sum of rewards from t to t+N-1 plus V(o_{t+N})
            # For success demos: r_t = -1 for all non-terminal steps
            steps_ahead = min(N, ep_length - 1 - i)
            n_step_reward = -steps_ahead  # sum of -1 for each step

            if i + N < ep_length:
                # Bootstrap with V(o_{t+N})
                _, bootstrap_value_idx, _ = frames[i + N]
                v_next = all_values[bootstrap_value_idx]
                # Normalize reward by max episode length (same scale as value)
                max_len = ep_length  # simplified: use episode length
                advantage = (n_step_reward / max_len) + v_next - v_t
            else:
                # Terminal: actual return from t to end
                # R_t = -(ep_length - 1 - frame_in_ep), normalized
                actual_return = -(ep_length - 1 - frame_in_ep) / ep_length
                advantage = actual_return - v_t

            all_advantages[value_idx] = advantage
            task_advantages[task_name].append((value_idx, advantage))

    logger.info(f"Advantage stats: mean={all_advantages.mean():.4f}, "
                f"std={all_advantages.std():.4f}, "
                f"min={all_advantages.min():.4f}, max={all_advantages.max():.4f}")

    # ── Binarize advantages with per-task threshold ───────────────────
    thresholds = {}
    binary_advantages = np.zeros(len(vf_dataset), dtype=np.int32)

    for task_name, task_advs in task_advantages.items():
        advs = np.array([a for _, a in task_advs])
        threshold = np.percentile(advs, args.positive_percentile)
        thresholds[task_name] = float(threshold)

        positive_count = 0
        for value_idx, adv in task_advs:
            is_positive = int(adv > threshold)
            binary_advantages[value_idx] = is_positive
            positive_count += is_positive

        pct = 100 * positive_count / len(task_advs)
        logger.info(f"Task '{task_name}': threshold={threshold:.4f}, "
                    f"positive={positive_count}/{len(task_advs)} ({pct:.1f}%)")

    # ── Save results ──────────────────────────────────────────────────
    result = {
        "config": {
            "value_checkpoint": args.value_checkpoint,
            "n_lookahead": args.n_lookahead,
            "positive_percentile": args.positive_percentile,
            "dataset_list_file": args.dataset_list_file,
        },
        "per_task_thresholds": thresholds,
        "stats": {
            "total_frames": len(vf_dataset),
            "total_positive": int(binary_advantages.sum()),
            "total_negative": int(len(binary_advantages) - binary_advantages.sum()),
            "advantage_mean": float(all_advantages.mean()),
            "advantage_std": float(all_advantages.std()),
        },
        # Store per-frame data: map from (dataset_repo_id, global_frame_idx) → advantage info
        "frames": [],
    }

    for i, (ds_idx, global_frame_idx, bin_idx, return_val) in enumerate(vf_dataset.frame_map):
        ds = vf_dataset.sub_datasets[ds_idx]
        result["frames"].append({
            "dataset": ds.repo_id,
            "global_idx": global_frame_idx,
            "advantage": float(all_advantages[i]),
            "is_positive": int(binary_advantages[i]),
            "value_pred": float(all_values[i]),
            "return_target": float(return_val),
        })

    with open(args.output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Saved advantages to {args.output_file}")
    logger.info(f"Summary: {result['stats']}")


if __name__ == "__main__":
    main()
