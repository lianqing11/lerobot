#!/usr/bin/env python3
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Compute offline metrics on the training dataset for a Pi0.5 (or compatible) checkpoint.

Loads `train_config.json` next to `model.safetensors`, rebuilds the same dataset as training,
and runs `policy.forward` (flow-matching loss) in eval mode — equivalent to training loss without backward.

Example:
  python examples/training/infer_pi05_trainset.py \\
    --checkpoint ckpt_xlerobot/newenv/_20260329_040424/checkpoints/006000/pretrained_model \\
    --max_batches 200
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging


def main() -> None:
    init_logging()
    register_third_party_plugins()
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser(description="Offline train-set inference for Pi0.5 checkpoints.")
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to .../pretrained_model containing train_config.json and model.safetensors",
    )
    p.add_argument("--batch_size", type=int, default=None, help="Override batch size (default: from train_config)")
    p.add_argument("--num_workers", type=int, default=None, help="Override dataloader workers")
    p.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Stop after this many batches (default: full pass over the dataset)",
    )
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--device", type=str, default=None, help="Override policy device, e.g. cuda or cpu")
    args = p.parse_args()

    ckpt_dir = args.checkpoint.resolve()
    train_cfg_path = ckpt_dir / "train_config.json"
    if not train_cfg_path.is_file():
        raise FileNotFoundError(f"Missing {train_cfg_path}")
    if not (ckpt_dir / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing {ckpt_dir / 'model.safetensors'}")

    cfg = TrainPipelineConfig.from_pretrained(str(train_cfg_path))
    cfg.policy.pretrained_path = ckpt_dir
    cfg.policy.compile_model = False
    if args.device is not None:
        cfg.policy.device = args.device
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    device = get_safe_torch_device(cfg.policy.device, log=True)
    set_seed(args.seed)

    logging.info("Building dataset (same as training config)...")
    dataset = make_dataset(cfg)

    if hasattr(cfg.policy, "drop_n_last_frames"):
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    logging.info("Loading policy weights from %s", ckpt_dir)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor_overrides = {
        "device_processor": {"device": str(policy.config.device)},
        "rename_observations_processor": {"rename_map": cfg.rename_map},
    }
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides=preprocessor_overrides,
    )

    use_amp = bool(getattr(cfg.policy, "use_amp", False))
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" and not use_amp and str(getattr(cfg.policy, "dtype", "")) == "bfloat16"
        else nullcontext()
    )

    sum_loss = 0.0
    n_batches = 0
    n_samples = 0

    limit = args.max_batches
    total = len(loader) if limit is None else min(limit, len(loader))
    bar = tqdm(loader, total=total, desc="train-set forward")
    with torch.no_grad():
        for batch in bar:
            batch = preprocessor(batch)
            with autocast_ctx:
                loss, out = policy.forward(batch)
            bs = next(iter(batch.values())).shape[0] if batch else 0
            sum_loss += float(loss.detach()) * bs
            n_samples += bs
            n_batches += 1
            bar.set_postfix(loss=float(loss.detach()), avg=sum_loss / max(n_samples, 1))
            if limit is not None and n_batches >= limit:
                break

    mean_loss = sum_loss / max(n_samples, 1)
    summary = {
        "checkpoint": str(ckpt_dir),
        "mean_loss": mean_loss,
        "n_batches": n_batches,
        "n_samples": n_samples,
        "batch_size": cfg.batch_size,
        "dataset_num_frames": dataset.num_frames,
        "dataset_num_episodes": dataset.num_episodes,
    }
    out_path = ckpt_dir / "trainset_infer_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Mean train-set forward loss: %.6f (%d samples in %d batches)", mean_loss, n_samples, n_batches)
    logging.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
