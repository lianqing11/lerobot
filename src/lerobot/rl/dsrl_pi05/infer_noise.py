#!/usr/bin/env python

import argparse
from pathlib import Path

import torch

from lerobot.rl.dsrl_pi05.dataset import Pi05SidecarDataset, move_batch_to_device
from lerobot.rl.dsrl_pi05.model import LatentSAC


def repeat_latent_noise(latent_noise: torch.Tensor, chunk_size: int = 50) -> torch.Tensor:
    """Convert [B, 32] or [32] latent_noise to Pi0.5 noise [B, 50, 32]."""
    if latent_noise.ndim == 1:
        latent_noise = latent_noise.unsqueeze(0)
    return latent_noise.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()


def predict_action_chunk_with_latent_noise(
    pi05_policy,
    batch: dict[str, torch.Tensor],
    latent_noise: torch.Tensor,
    chunk_size: int = 50,
    **kwargs,
) -> torch.Tensor:
    """Call Pi0.5 with actor-produced latent_noise repeated across the chunk."""
    noise = repeat_latent_noise(latent_noise, chunk_size=chunk_size).to(latent_noise.device)
    return pi05_policy.predict_action_chunk(batch, noise=noise, **kwargs)


def load_latent_sac(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> LatentSAC:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint["args"]
    model = LatentSAC(
        obs_state_dim=int(args["obs_state_dim"]) if "obs_state_dim" in args else 13,
        latent_noise_dim=int(args["latent_noise_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        feature_dim=int(args["feature_dim"]),
        noise_bound=float(args["noise_bound"]),
        resnet_weights=str(args["resnet_weights"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


@torch.no_grad()
def select_latent_noise(model: LatentSAC, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    feature = model.encode_obs(batch)
    return model.actor.select(feature)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a latent actor on one sidecar observation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_latent_sac(args.checkpoint, device=device)
    dataset = Pi05SidecarDataset(args.dataset_root, image_size=args.image_size)
    sample = dataset[args.index]
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
        if key in {"obs_state", "image_main", "image_wrist"}
    }
    batch = move_batch_to_device(batch, device)
    latent_noise = select_latent_noise(model, batch)
    repeated_noise = repeat_latent_noise(latent_noise)
    print(f"latent_noise shape={tuple(latent_noise.shape)} min={latent_noise.min().item():.4f} max={latent_noise.max().item():.4f}")
    print(f"repeated_noise shape={tuple(repeated_noise.shape)}")


if __name__ == "__main__":
    main()

