#!/usr/bin/env python
"""
Train a distributional value function for RECAP-style advantage-conditioned RL.

Based on: "π*0.6: a VLA That Learns From Experience" (Physical Intelligence)

The value function predicts V(o_t, ℓ) — the (negative) normalized number of
steps to success — as a categorical distribution over 201 bins in [-1, 0].

Supports multi-GPU via `accelerate`:
    # Single GPU
    python train_value_function.py --dataset_list_file ... --output_dir ...

    # Multi-GPU
    accelerate launch --num_processes 4 train_value_function.py --dataset_list_file ... --output_dir ...

All episodes in the dataset_list_file are assumed to be successful demonstrations.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.utils.data import DataLoader, Dataset, Subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset list file parsing ──────────────────────────────────────────────


def parse_dataset_list_file(filepath: str) -> list[tuple[str, str]]:
    """Parse dataset list file, return list of (repo_id, root_path)."""
    entries = []
    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                path = parts[0].rstrip("/")
                repo_id = Path(path).name
                entries.append((repo_id, path))
            elif len(parts) == 2:
                entries.append((parts[0], parts[1].rstrip("/")))
            else:
                raise ValueError(f"Invalid line: '{raw_line}'")
    return entries


# ── Value function dataset ─────────────────────────────────────────────────


class ValueFunctionDataset(Dataset):
    """Wraps multiple LeRobotDatasets and adds per-frame return targets.

    Each sample contains all fields from the underlying LeRobotDataset plus:
        - 'value_target_bin': int, discretized return bin index in [0, 200]
        - 'value_target': float, normalized return in [-1, 0]
    """

    def __init__(
        self,
        dataset_entries: list[tuple[str, str]],
        num_bins: int = 201,
        all_success: bool = True,
        video_backend: str = "pyav",
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.num_bins = num_bins
        self.all_success = all_success
        self.sub_datasets = []
        self.episode_infos = []

        logger.info(f"Loading {len(dataset_entries)} datasets...")

        max_length_per_task = {}
        for repo_id, root in dataset_entries:
            ds = LeRobotDataset(repo_id, root=root, video_backend=video_backend)
            ds_idx = len(self.sub_datasets)
            self.sub_datasets.append(ds)

            task_name = None
            if ds.meta.tasks is not None and len(ds.meta.tasks) > 0:
                task_name = str(ds.meta.tasks.index[0])

            for ep in ds.meta.episodes:
                ep_idx = ep["episode_index"]
                length = ep["length"]
                if task_name not in max_length_per_task:
                    max_length_per_task[task_name] = length
                else:
                    max_length_per_task[task_name] = max(max_length_per_task[task_name], length)
                self.episode_infos.append((ds_idx, ep_idx, length, task_name))

        self.frame_map = []

        for ds_idx, ep_idx, ep_length, task_name in self.episode_infos:
            ds = self.sub_datasets[ds_idx]
            max_len = max_length_per_task.get(task_name, ep_length)

            for frame_in_ep in range(ep_length):
                steps_remaining = ep_length - 1 - frame_in_ep
                raw_return = -steps_remaining
                normalized_return = raw_return / max_len
                normalized_return = max(-1.0, min(0.0, normalized_return))

                bin_idx = int((normalized_return + 1.0) * (self.num_bins - 1))
                bin_idx = max(0, min(self.num_bins - 1, bin_idx))

                ep_data = ds.meta.episodes[ep_idx]
                global_idx = ep_data["dataset_from_index"] + frame_in_ep
                self.frame_map.append((ds_idx, global_idx, bin_idx, normalized_return))

        logger.info(
            f"Total frames: {len(self.frame_map)}, "
            f"Datasets: {len(self.sub_datasets)}, "
            f"Episodes: {len(self.episode_infos)}, "
            f"Tasks: {list(max_length_per_task.keys())}"
        )

    def __len__(self):
        return len(self.frame_map)

    def __getitem__(self, idx):
        ds_idx, global_frame_idx, bin_idx, return_value = self.frame_map[idx]
        item = self.sub_datasets[ds_idx][global_frame_idx]
        item["value_target_bin"] = bin_idx
        item["value_target"] = return_value
        return item


# ── Collate / tokenizer helpers ────────────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        elif isinstance(vals[0], (int, float)):
            collated[k] = torch.tensor(vals)
        elif isinstance(vals[0], str):
            collated[k] = vals
        else:
            collated[k] = vals
    return collated


def prepare_tokenized_input(tasks, tokenizer, max_length=200, device="cuda"):
    encoded = tokenizer(
        tasks, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt",
    )
    tokens = encoded["input_ids"].to(device)
    masks = encoded["attention_mask"].to(device).bool()
    return tokens, masks


# ── Evaluation ─────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, dataloader, image_keys, tokenizer, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_samples = 0

    # unwrap DDP/FSDP to call custom methods
    raw_model = model.module if hasattr(model, "module") else model

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, img_masks = raw_model.preprocess_images(batch, image_keys)
        tasks = batch.get("task", [""] * len(batch["value_target_bin"]))
        tokens, masks = prepare_tokenized_input(tasks, tokenizer, device=str(device))
        target_bins = batch["value_target_bin"].long().to(device)

        loss, info = raw_model.compute_loss(images, img_masks, tokens, masks, target_bins)

        bs = target_bins.shape[0]
        total_loss += info["loss"] * bs
        total_mae += info["mae"] * bs
        total_samples += bs

    model.train()
    if total_samples == 0:
        return {"eval_loss": 0.0, "eval_mae": 0.0, "eval_samples": 0}

    return {
        "eval_loss": total_loss / total_samples,
        "eval_mae": total_mae / total_samples,
        "eval_samples": total_samples,
    }


# ── Training ───────────────────────────────────────────────────────────────


def train(args):
    # ── Accelerator (handles single-GPU, multi-GPU, mixed precision) ──
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="bf16" if args.precision == "bfloat16" else "no",
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False,
    )
    is_main = accelerator.is_main_process
    device = accelerator.device

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(Path(args.output_dir) / "train_config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    # ── Tokenizer ─────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset (load on main first to avoid race conditions) ─────────
    if is_main:
        logger.info("Loading datasets...")
    dataset_entries = parse_dataset_list_file(args.dataset_list_file)
    vf_dataset = ValueFunctionDataset(
        dataset_entries, num_bins=201, all_success=True, video_backend=args.video_backend,
    )

    eval_size = min(args.eval_episodes * 500, len(vf_dataset) // 5)
    train_size = len(vf_dataset) - eval_size
    indices = list(range(len(vf_dataset)))
    train_subset = Subset(vf_dataset, indices[:train_size])
    eval_subset = Subset(vf_dataset, indices[train_size:])
    if is_main:
        logger.info(f"Train: {train_size} frames, Eval: {eval_size} frames")

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    sample = vf_dataset[0]
    image_keys = [k for k in sample.keys() if "image" in k and isinstance(sample[k], torch.Tensor)]
    if is_main:
        logger.info(f"Image features: {image_keys}")

    # ── Model ─────────────────────────────────────────────────────────
    from lerobot.policies.pi05.value_function_pi05 import PI05ValueFunction

    model = PI05ValueFunction(
        vlm_variant=args.vlm_variant, image_resolution=(224, 224),
        precision="float32",  # accelerator handles mixed precision
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.init_from_policy:
        if is_main:
            logger.info(f"Initializing VLM from policy: {args.init_from_policy}")
        _load_vlm_weights_from_policy(model, args.init_from_policy)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        logger.info(f"Params: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.1,
    )

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"]
        for _ in range(start_step):
            scheduler.step()
        if is_main:
            logger.info(f"Resumed from step {start_step}")

    # ── Wrap with accelerator ─────────────────────────────────────────
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler,
    )

    # ── Wandb (main process only) ─────────────────────────────────────
    wandb_run = None
    if args.wandb_project and is_main:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name,
            config=vars(args), resume="allow",
        )

    # ── Training loop ─────────────────────────────────────────────────
    model.train()
    data_iter = iter(train_loader)
    running_loss = 0.0
    running_mae = 0.0
    log_count = 0
    train_start = time.time()

    raw_model = accelerator.unwrap_model(model)

    if is_main:
        logger.info(
            f"Training for {args.steps} steps (from {start_step}), "
            f"batch_size={args.batch_size} x {accelerator.num_processes} GPUs"
        )

    for global_step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images, img_masks = raw_model.preprocess_images(batch, image_keys)
        tasks = batch.get("task", [""] * args.batch_size)
        tokens, masks = prepare_tokenized_input(tasks, tokenizer, device=str(device))
        target_bins = batch["value_target_bin"].long().to(device)

        with accelerator.autocast():
            loss, info = raw_model.compute_loss(images, img_masks, tokens, masks, target_bins)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if args.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        running_loss += info["loss"]
        running_mae += info["mae"]
        log_count += 1

        # ── Log (main only) ───────────────────────────────────────
        if global_step % args.log_freq == 0 and is_main:
            avg_loss = running_loss / log_count
            avg_mae = running_mae / log_count
            elapsed = time.time() - train_start
            steps_done = global_step - start_step
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

            logger.info(
                f"step {global_step}/{args.steps} | "
                f"loss={avg_loss:.4f} | mae={avg_mae:.4f} | "
                f"pred_v={info['pred_value_mean']:.4f} | tgt_v={info['target_value_mean']:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {steps_per_sec:.2f} it/s"
            )
            if wandb_run:
                wandb_run.log({
                    "train/loss": avg_loss, "train/mae": avg_mae,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/steps_per_sec": steps_per_sec,
                }, step=global_step)

            running_loss = 0.0
            running_mae = 0.0
            log_count = 0

        # ── Eval (main only) ──────────────────────────────────────
        if args.eval_freq > 0 and global_step % args.eval_freq == 0 and is_main:
            eval_metrics = evaluate(model, eval_loader, image_keys, tokenizer, device, max_batches=50)
            logger.info(
                f"[Eval] step {global_step} | "
                f"loss={eval_metrics['eval_loss']:.4f} | mae={eval_metrics['eval_mae']:.4f}"
            )
            if wandb_run:
                wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=global_step)

        # ── Save (main only) ──────────────────────────────────────
        if (global_step % args.save_freq == 0 or global_step == args.steps) and is_main:
            ckpt_path = Path(args.output_dir) / f"checkpoint_{global_step:06d}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": global_step,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "vlm_variant": args.vlm_variant,
                        "num_bins": 201,
                        "precision": args.precision,
                        "image_keys": image_keys,
                    },
                },
                ckpt_path / "value_function.pt",
            )
            logger.info(f"Saved checkpoint → {ckpt_path}")

        # Sync all processes at save boundaries
        if global_step % args.save_freq == 0:
            accelerator.wait_for_everyone()

    total_time = time.time() - train_start
    if is_main:
        logger.info(f"Done. {args.steps - start_step} steps in {total_time/60:.1f} min")
        if wandb_run:
            wandb_run.finish()


def _load_vlm_weights_from_policy(model, policy_path: str):
    """Load VLM (PaliGemma) weights from a pretrained PI05 policy checkpoint."""
    from safetensors.torch import load_file

    safetensors_path = Path(policy_path) / "model.safetensors"
    if not safetensors_path.exists():
        logger.warning(f"No model.safetensors at {policy_path}, skipping VLM init")
        return

    state_dict = load_file(str(safetensors_path))
    vlm_state = {}
    for key, val in state_dict.items():
        prefix = "paligemma_with_expert.paligemma."
        model_prefix = "model.paligemma_with_expert.paligemma."
        if key.startswith(model_prefix):
            new_key = "paligemma." + key[len(model_prefix):]
            vlm_state[new_key] = val
        elif key.startswith(prefix):
            new_key = "paligemma." + key[len(prefix):]
            vlm_state[new_key] = val

    if not vlm_state:
        logger.warning("No VLM weights found in policy checkpoint")
        return

    missing, unexpected = model.load_state_dict(vlm_state, strict=False)
    loaded = len(vlm_state) - len(unexpected)
    logger.info(f"Loaded {loaded} VLM weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="Train RECAP value function")
    # Data
    p.add_argument("--dataset_list_file", type=str, required=True)
    p.add_argument("--video_backend", type=str, default="pyav")
    # Model
    p.add_argument("--vlm_variant", type=str, default="gemma_300m", choices=["gemma_300m", "gemma_2b"])
    p.add_argument("--precision", type=str, default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--init_from_policy", type=str, default=None)
    p.add_argument("--gradient_checkpointing", action="store_true")
    # Training
    p.add_argument("--batch_size", type=int, default=16, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    # Logging
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_freq", type=int, default=1000)
    p.add_argument("--eval_freq", type=int, default=500)
    p.add_argument("--eval_episodes", type=int, default=5, help="Approx episodes held out for eval")
    p.add_argument("--output_dir", type=str, default="checkpoints/value_function")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
