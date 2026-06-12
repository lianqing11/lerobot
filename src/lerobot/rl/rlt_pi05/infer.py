#!/usr/bin/env python

import argparse
from pathlib import Path

import torch

from lerobot.rl.rlt_pi05.dataset import RLTSidecarDataset, move_batch_to_device
from lerobot.rl.rlt_pi05.model import RLTPolicy


def load_rlt_policy(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> tuple[RLTPolicy, dict, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = RLTPolicy(
        obs_state_dim=int(args["obs_state_dim"]),
        proprio_hidden=int(args["proprio_hidden"]),
        feature_dim=int(args["feature_dim"]),
        resnet_weights=str(args["resnet_weights"]),
        chunk_C=int(args["chunk_C"]),
        action_dim=int(args["action_dim"]),
        hidden_dim=int(args["hidden_dim"]),
        hidden_layers=int(args["hidden_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, args, int(checkpoint["step"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test an RLT checkpoint on one sidecar observation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, ckpt_args, step = load_rlt_policy(args.checkpoint, device=device)
    dataset = RLTSidecarDataset(
        args.dataset_root,
        includes=args.include,
        chunk_C=int(ckpt_args["chunk_C"]),
        action_dim=int(ckpt_args["action_dim"]),
        image_size=int(ckpt_args["image_size"]),
    )
    sample = dataset[args.index]
    batch = {key: value.unsqueeze(0) for key, value in sample.items()}
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        feature = model.encode_obs(batch)
        mu = model.actor_mu(feature, batch["ref"])
        action = mu.view(-1, model.chunk_C, model.action_dim)

    print(f"checkpoint_step={step}")
    print(f"actor_output_shape={tuple(action.shape)}")
    print(
        f"action mean={action.mean().item():.4f} std={action.std().item():.4f} "
        f"min={action.min().item():.4f} max={action.max().item():.4f}"
    )


if __name__ == "__main__":
    main()

