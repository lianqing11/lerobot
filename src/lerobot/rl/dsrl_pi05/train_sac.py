#!/usr/bin/env python

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lerobot.rl.dsrl_pi05.dataset import Pi05SidecarDataset, move_batch_to_device
from lerobot.rl.dsrl_pi05.model import LatentSAC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-noise SAC on Pi0.5 rollout sidecars.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resnet-weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--noise-reduction", choices=["first"], default="first")
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=512)
    parser.add_argument("--latent-noise-dim", type=int, default=32)
    parser.add_argument("--noise-bound", type=float, default=1.5)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-entropy", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Pi05SidecarDataset(
        args.dataset_root,
        image_size=args.image_size,
        noise_reduction=args.noise_reduction,
    )
    print(f"Loaded dataset: {dataset.summary}")
    if dataset.summary.reward_values == (0,):
        print("WARNING: all terminal rewards are 0; this run only verifies the training pipeline.")
    if dataset.summary.repeat_noise_max_diff > 1e-4:
        print(
            "WARNING: sidecar noise is full [50,32], not repeated [32]; "
            "using noise[0] as latent_noise because --noise-reduction=first."
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(dataset) >= args.batch_size,
        persistent_workers=args.num_workers > 0,
    )
    batch_iter = cycle(loader)

    model = LatentSAC(
        obs_state_dim=dataset.summary.obs_state_dim,
        latent_noise_dim=args.latent_noise_dim,
        hidden_dim=args.hidden_dim,
        feature_dim=args.feature_dim,
        noise_bound=args.noise_bound,
        resnet_weights=args.resnet_weights,
    ).to(device)

    actor_optim = torch.optim.Adam(model.actor.parameters(), lr=args.actor_lr)
    critic_optim = torch.optim.Adam(
        list(model.encoder.proprio_encoder.parameters())
        + list(model.encoder.fusion.parameters())
        + list(model.critic1.parameters())
        + list(model.critic2.parameters()),
        lr=args.critic_lr,
    )
    alpha_optim = torch.optim.Adam([model.log_alpha], lr=args.alpha_lr)

    train_start_time = time.perf_counter()
    last_log_time = train_start_time
    last_log_step = 0

    for step in range(1, args.steps + 1):
        batch = move_batch_to_device(next(batch_iter), device)

        feature = model.encode_obs(batch)
        with torch.no_grad():
            next_feature = model.encode_obs(batch, next_obs=True)
            next_latent_noise, next_log_prob = model.actor.sample(next_feature)
            target_q1 = model.target_critic1(next_feature, next_latent_noise)
            target_q2 = model.target_critic2(next_feature, next_latent_noise)
            alpha = model.log_alpha.exp()
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
            target = batch["reward"] + args.gamma * (1.0 - batch["done"]) * target_q

        q1 = model.critic1(feature, batch["latent_noise"])
        q2 = model.critic2(feature, batch["latent_noise"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_optim.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_optim.step()

        feature = model.encode_obs(batch)
        latent_noise, log_prob = model.actor.sample(feature)
        q_pi = torch.min(model.critic1(feature, latent_noise), model.critic2(feature, latent_noise))
        alpha = model.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q_pi).mean()
        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        alpha_loss = -(model.log_alpha.exp() * (log_prob + args.target_entropy).detach()).mean()
        alpha_optim.zero_grad(set_to_none=True)
        alpha_loss.backward()
        alpha_optim.step()

        model.update_targets(args.tau)

        if step == 1 or step % args.log_every == 0:
            now = time.perf_counter()
            interval_steps = step - last_log_step
            interval_time = now - last_log_time
            iter_time = interval_time / max(interval_steps, 1)
            elapsed = now - train_start_time
            eta = iter_time * max(args.steps - step, 0)
            print(
                f"step={step} critic_loss={critic_loss.item():.4f} "
                f"actor_loss={actor_loss.item():.4f} alpha={model.log_alpha.exp().item():.4f} "
                f"reward_mean={batch['reward'].mean().item():.4f} "
                f"iter_time={iter_time:.3f}s elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}"
            )
            last_log_time = now
            last_log_step = step
        if step % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_{step:06d}.pt", model, args, step, dataset.summary.obs_state_dim)

    save_checkpoint(output_dir / "checkpoint_last.pt", model, args, args.steps, dataset.summary.obs_state_dim)


def save_checkpoint(
    path: Path,
    model: LatentSAC,
    args: argparse.Namespace,
    step: int,
    obs_state_dim: int,
) -> None:
    saved_args = vars(args).copy()
    saved_args["obs_state_dim"] = obs_state_dim
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
