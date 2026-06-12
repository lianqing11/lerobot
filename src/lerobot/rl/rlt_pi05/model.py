#!/usr/bin/env python

from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.rl.dsrl_pi05.model import ObservationEncoder, mlp


class RLTActor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        ref_dim: int,
        action_dim_flat: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.net = mlp(feature_dim + ref_dim, hidden_dim, action_dim_flat, hidden_layers=hidden_layers)

    def forward(self, feature: Tensor, ref_flat: Tensor) -> Tensor:
        return self.net(torch.cat([feature, ref_flat], dim=-1))


class RLTCritic(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim_flat: int,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        self.net = mlp(feature_dim + action_dim_flat, hidden_dim, 1, hidden_layers=hidden_layers)

    def forward(self, feature: Tensor, action_flat: Tensor) -> Tensor:
        return self.net(torch.cat([feature, action_flat], dim=-1))


class RLTPolicy(nn.Module):
    def __init__(
        self,
        *,
        obs_state_dim: int = 13,
        proprio_hidden: int = 256,
        feature_dim: int = 256,
        resnet_weights: str = "imagenet",
        chunk_C: int = 16,
        action_dim: int = 7,
        hidden_dim: int = 256,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        action_dim_flat = chunk_C * action_dim
        self.chunk_C = chunk_C
        self.action_dim = action_dim
        self.encoder = ObservationEncoder(
            obs_state_dim=obs_state_dim,
            proprio_hidden_dim=proprio_hidden,
            feature_dim=feature_dim,
            resnet_weights=resnet_weights,
        )
        self.actor = RLTActor(feature_dim, action_dim_flat, action_dim_flat, hidden_dim, hidden_layers)
        self.critic1 = RLTCritic(feature_dim, action_dim_flat, hidden_dim, hidden_layers)
        self.critic2 = RLTCritic(feature_dim, action_dim_flat, hidden_dim, hidden_layers)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)

    def encode_obs(self, batch: dict[str, Tensor], next_obs: bool = False) -> Tensor:
        prefix = "next_" if next_obs else ""
        return self.encoder(
            batch[f"{prefix}obs_state"],
            batch[f"{prefix}image_main"],
            batch[f"{prefix}image_wrist"],
        )

    def flatten_action(self, action: Tensor) -> Tensor:
        return action.reshape(action.shape[0], self.chunk_C * self.action_dim)

    def actor_mu(self, feature: Tensor, ref: Tensor) -> Tensor:
        ref_flat = self.flatten_action(ref)
        return self.actor(feature, ref_flat)

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        for target, source in (
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.lerp_(source_param.data, tau)

