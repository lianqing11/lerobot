#!/usr/bin/env python

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lerobot.rl.rlt_pi05.dataset import RLTSidecarDataset, move_batch_to_device
from lerobot.rl.rlt_pi05.model import RLTPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RLT action-prefix actor-critic on Pi0.5 sidecars.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--dataset-list", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-suffix", default="")
    parser.add_argument("--load-from", default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--resnet-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--obs-state-dim", type=int, default=13)
    parser.add_argument("--proprio-hidden", type=int, default=256)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--chunk-C", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--action-std", type=float, default=0.05)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--ref-dropout-p", type=float, default=0.5)
    parser.add_argument("--bc-lambda-init", type=float, default=1.0)
    parser.add_argument("--bc-lambda-final", type=float, default=0.01)
    parser.add_argument("--bc-anneal-steps", type=int, default=5000)
    parser.add_argument("--bc-target", choices=["action", "ref"], default="action")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def cycle(loader: DataLoader, sampler: DistributedSampler | None = None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
            epoch += 1
        for batch in loader:
            yield batch


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    distributed, local_rank, rank, world_size = init_distributed(args)
    device = torch.device("cuda", local_rank) if distributed else torch.device(args.device)

    output_dir = Path(args.output_dir)
    if rank == 0 and not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset = RLTSidecarDataset(
        args.dataset_root,
        includes=args.include,
        dataset_list=args.dataset_list,
        chunk_C=args.chunk_C,
        action_dim=args.action_dim,
        image_size=args.image_size,
    )
    if rank == 0:
        print(f"Loaded dataset: {dataset.summary}")
        if distributed:
            print(f"Distributed training: world_size={world_size} per_rank_batch_size={args.batch_size}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size * world_size,
        persistent_workers=args.num_workers > 0,
    )
    batch_iter = cycle(loader, sampler)

    model = RLTPolicy(
        obs_state_dim=dataset.summary.obs_state_dim,
        proprio_hidden=args.proprio_hidden,
        feature_dim=args.feature_dim,
        resnet_weights=args.resnet_weights,
        chunk_C=args.chunk_C,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)
    start_step = 0
    if args.load_from is not None:
        start_step = load_model_checkpoint(model, args.load_from, device)
        if rank == 0:
            print(f"Loaded model weights from {args.load_from} at checkpoint_step={start_step}")
    modules = TrainModules(model, distributed=distributed, local_rank=local_rank)

    actor_optim = torch.optim.Adam(model.actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(
        list(model.encoder.proprio_encoder.parameters())
        + list(model.encoder.fusion.parameters())
        + list(model.critic1.parameters())
        + list(model.critic2.parameters()),
        lr=args.critic_lr,
    )

    if args.dry_run:
        batch = move_batch_to_device(next(batch_iter), device)
        metrics = compute_losses(modules, batch, args, step=0)
        if rank == 0:
            print_batch_shapes(batch)
            print_metrics("dry_run", metrics)
        cleanup_distributed(distributed)
        return

    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = 0

    for step in range(1, args.steps + 1):
        batch = move_batch_to_device(next(batch_iter), device)
        global_step = start_step + step
        metrics = train_step(modules, batch, args, step, actor_optim, critic_optim)

        if rank == 0 and (step == 1 or step % args.log_every == 0):
            now = time.perf_counter()
            interval_steps = step - last_log_step
            interval_time = now - last_log_time
            iter_time = interval_time / max(interval_steps, 1)
            elapsed = now - train_start_time
            eta = iter_time * max(args.steps - step, 0)
            print_metrics(
                f"step={global_step}",
                {
                    **metrics,
                    "iter_time": iter_time,
                    "elapsed_s": elapsed,
                    "eta_s": eta,
                },
            )
            last_log_time = now
            last_log_step = step

        if step % args.save_every == 0:
            if rank == 0:
                save_checkpoint(
                    output_dir / checkpoint_filename(global_step, args.checkpoint_suffix),
                    model,
                    args,
                    global_step,
                    dataset.summary.obs_state_dim,
                )
            if distributed:
                dist.barrier()

    if rank == 0:
        save_checkpoint(
            output_dir / checkpoint_filename(None, args.checkpoint_suffix),
            model,
            args,
            start_step + args.steps,
            dataset.summary.obs_state_dim,
        )
    cleanup_distributed(distributed)


@dataclass
class TrainModules:
    model: RLTPolicy
    distributed: bool = False
    local_rank: int = 0

    def __post_init__(self) -> None:
        if self.distributed:
            self.encoder = DDP(self.model.encoder, device_ids=[self.local_rank])
            self.actor = DDP(self.model.actor, device_ids=[self.local_rank])
            self.critic1 = DDP(self.model.critic1, device_ids=[self.local_rank])
            self.critic2 = DDP(self.model.critic2, device_ids=[self.local_rank])
        else:
            self.encoder = self.model.encoder
            self.actor = self.model.actor
            self.critic1 = self.model.critic1
            self.critic2 = self.model.critic2


def init_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    torch.manual_seed(args.seed + rank)
    return True, local_rank, rank, world_size


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def flatten_action(model: RLTPolicy, action: Tensor) -> Tensor:
    return action.reshape(action.shape[0], model.chunk_C * model.action_dim)


def encode_obs(modules: TrainModules, batch: dict[str, Tensor], next_obs: bool = False) -> Tensor:
    prefix = "next_" if next_obs else ""
    return modules.encoder(
        batch[f"{prefix}obs_state"],
        batch[f"{prefix}image_main"],
        batch[f"{prefix}image_wrist"],
    )


def actor_mu(modules: TrainModules, feature: Tensor, ref: Tensor) -> Tensor:
    return modules.actor(feature, flatten_action(modules.model, ref))


def compute_losses(modules: TrainModules, batch: dict[str, Tensor], args: argparse.Namespace, step: int) -> dict[str, Tensor]:
    metrics = compute_critic_metrics(modules, batch, args)
    metrics.update(compute_actor_metrics(modules, batch, args, step))
    return metrics


def compute_critic_metrics(modules: TrainModules, batch: dict[str, Tensor], args: argparse.Namespace) -> dict[str, Tensor]:
    model = modules.model
    action_flat = flatten_action(model, batch["action"])
    feature = encode_obs(modules, batch)

    with torch.no_grad():
        next_feature = encode_obs(modules, batch, next_obs=True)
        next_mu = actor_mu(modules, next_feature, batch["next_ref"])
        next_action = next_mu + args.action_std * torch.randn_like(next_mu)
        target_q1 = model.target_critic1(next_feature, next_action)
        target_q2 = model.target_critic2(next_feature, next_action)
        target_q = torch.min(target_q1, target_q2)
        target = batch["reward"] + args.gamma * (1.0 - batch["done"]) * target_q

    q1 = modules.critic1(feature, action_flat)
    q2 = modules.critic2(feature, action_flat)
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    return {
        "critic_loss": critic_loss,
        "reward_mean": batch["reward"].mean(),
        "done_mean": batch["done"].mean(),
    }


def compute_actor_metrics(modules: TrainModules, batch: dict[str, Tensor], args: argparse.Namespace, step: int) -> dict[str, Tensor]:
    model = modules.model
    action_flat = flatten_action(model, batch["action"])
    actor_feature = encode_obs(modules, batch)
    actor_ref = apply_ref_dropout(batch["ref"], args.ref_dropout_p)
    mu = actor_mu(modules, actor_feature, actor_ref)
    actor_q = torch.min(
        modules.critic1(actor_feature.detach(), mu),
        modules.critic2(actor_feature.detach(), mu),
    )
    bc_lambda = get_bc_lambda(args, step)
    bc_target = action_flat if args.bc_target == "action" else flatten_action(model, batch["ref"])
    bc_loss = F.mse_loss(mu, bc_target)
    actor_loss = -actor_q.mean() + bc_lambda * bc_loss

    with torch.no_grad():
        ref_flat = flatten_action(model, batch["ref"])
        mu_with_ref = actor_mu(modules, actor_feature, batch["ref"])
        mu_ref_delta = mu_with_ref - ref_flat
        mu_ref_l2 = mu_ref_delta.norm(dim=-1).mean()
        mu_ref_rms = mu_ref_delta.square().mean(dim=-1).sqrt().mean()
        denom = mu_with_ref.norm(dim=-1).clamp_min(1e-6) * ref_flat.norm(dim=-1).clamp_min(1e-6)
        cos_sim = ((mu_with_ref * ref_flat).sum(dim=-1) / denom).mean()

    return {
        "actor_loss": actor_loss,
        "bc_loss": bc_loss,
        "bc_lambda": torch.tensor(bc_lambda, device=mu.device),
        "q_mean": actor_q.mean(),
        "mu_abs_mean": mu.abs().mean(),
        "mu_ref_l2": mu_ref_l2,
        "mu_ref_rms": mu_ref_rms,
        "action_abs_mean": action_flat.abs().mean(),
        "ref_mu_cos": cos_sim,
    }


def train_step(
    modules: TrainModules,
    batch: dict[str, Tensor],
    args: argparse.Namespace,
    step: int,
    actor_optim: torch.optim.Optimizer,
    critic_optim: torch.optim.Optimizer,
) -> dict[str, float]:
    metrics = compute_critic_metrics(modules, batch, args)

    critic_optim.zero_grad(set_to_none=True)
    metrics["critic_loss"].backward()
    critic_optim.step()

    if step > args.warmup_steps:
        actor_metrics = compute_actor_metrics(modules, batch, args, step)
        actor_optim.zero_grad(set_to_none=True)
        actor_metrics["actor_loss"].backward()
        actor_optim.step()
    else:
        with torch.no_grad():
            actor_metrics = compute_actor_metrics(modules, batch, args, step)

    modules.model.update_targets(args.tau)
    metrics.update(actor_metrics)
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def apply_ref_dropout(ref: Tensor, dropout_p: float) -> Tensor:
    if dropout_p <= 0.0:
        return ref
    keep = torch.rand(ref.shape[0], 1, 1, device=ref.device) >= dropout_p
    return ref * keep.to(ref.dtype)


def get_bc_lambda(args: argparse.Namespace, step: int) -> float:
    if args.bc_anneal_steps <= 0:
        return args.bc_lambda_final
    ratio = min(max(step, 0), args.bc_anneal_steps) / args.bc_anneal_steps
    return args.bc_lambda_init + ratio * (args.bc_lambda_final - args.bc_lambda_init)


def load_model_checkpoint(model: RLTPolicy, checkpoint_path: str | Path, device: torch.device) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return int(checkpoint["step"])


def print_batch_shapes(batch: dict[str, Tensor]) -> None:
    shapes = {key: tuple(value.shape) for key, value in batch.items()}
    print(f"Batch shapes: {shapes}")


def print_metrics(prefix: str, metrics: dict[str, Tensor | float]) -> None:
    parts = [prefix]
    for key, value in metrics.items():
        if isinstance(value, Tensor):
            value = float(value.detach().cpu())
        if key.endswith("_s"):
            parts.append(f"{key}={format_seconds(value)}")
        else:
            parts.append(f"{key}={value:.4f}")
    print(" ".join(parts))


def checkpoint_filename(step: int | None, suffix: str) -> str:
    suffix_part = f"_{suffix}" if suffix else ""
    if step is None:
        return f"checkpoint_last{suffix_part}.pt"
    return f"checkpoint_{step:06d}{suffix_part}.pt"


def save_checkpoint(
    path: Path,
    model: RLTPolicy,
    args: argparse.Namespace,
    step: int,
    obs_state_dim: int,
) -> None:
    saved_args = vars(args).copy()
    saved_args.update(
        {
            "obs_state_dim": obs_state_dim,
            "proprio_hidden": args.proprio_hidden,
            "feature_dim": args.feature_dim,
            "resnet_weights": args.resnet_weights,
            "image_size": args.image_size,
            "chunk_C": args.chunk_C,
            "action_dim": args.action_dim,
            "action_std": args.action_std,
        }
    )
    torch.save(
        {
            "step": step,
            "args": saved_args,
            "model": model.state_dict(),
        },
        path,
    )
    print(f"Saved checkpoint to {path}")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
