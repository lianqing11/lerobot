#!/usr/bin/env python

from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal
from torchvision.models import ResNet50_Weights, resnet50


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def mlp(input_dim: int, hidden_dim: int, output_dim: int, hidden_layers: int = 3) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()])
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


class SharedResNet50Encoder(nn.Module):
    def __init__(self, weights: str = "imagenet") -> None:
        super().__init__()
        if weights == "imagenet":
            weight_enum = ResNet50_Weights.IMAGENET1K_V2
        elif weights == "none":
            weight_enum = None
        else:
            raise ValueError("--resnet-weights must be 'imagenet' or 'none'")

        backbone = resnet50(weights=weight_enum)
        self.output_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, image_main: Tensor, image_wrist: Tensor) -> Tensor:
        main = self.backbone(image_main)
        wrist = self.backbone(image_wrist)
        return torch.cat([main, wrist], dim=-1)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        obs_state_dim: int,
        proprio_hidden_dim: int = 256,
        feature_dim: int = 512,
        resnet_weights: str = "imagenet",
    ) -> None:
        super().__init__()
        self.image_encoder = SharedResNet50Encoder(weights=resnet_weights)
        self.proprio_encoder = mlp(obs_state_dim, proprio_hidden_dim, proprio_hidden_dim, hidden_layers=1)
        fusion_dim = self.image_encoder.output_dim * 2 + proprio_hidden_dim
        self.fusion = mlp(fusion_dim, feature_dim, feature_dim, hidden_layers=1)
        self.output_dim = feature_dim

    def forward(self, obs_state: Tensor, image_main: Tensor, image_wrist: Tensor) -> Tensor:
        image_feature = self.image_encoder(image_main, image_wrist)
        proprio_feature = self.proprio_encoder(obs_state)
        return self.fusion(torch.cat([image_feature, proprio_feature], dim=-1))


class LatentActor(nn.Module):
    def __init__(self, feature_dim: int, latent_noise_dim: int, hidden_dim: int, noise_bound: float) -> None:
        super().__init__()
        self.noise_bound = noise_bound
        self.net = mlp(feature_dim, hidden_dim, latent_noise_dim * 2)

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self.net(feature).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(feature)
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        squashed = torch.tanh(raw)
        latent_noise = squashed * self.noise_bound
        log_prob = dist.log_prob(raw) - torch.log(self.noise_bound * (1 - squashed.pow(2)) + 1e-6)
        return latent_noise, log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def select(self, feature: Tensor) -> Tensor:
        mean, _ = self(feature)
        return torch.tanh(mean) * self.noise_bound


class Critic(nn.Module):
    def __init__(self, feature_dim: int, latent_noise_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = mlp(feature_dim + latent_noise_dim, hidden_dim, 1)

    def forward(self, feature: Tensor, latent_noise: Tensor) -> Tensor:
        return self.net(torch.cat([feature, latent_noise], dim=-1))


class LatentSAC(nn.Module):
    def __init__(
        self,
        obs_state_dim: int,
        latent_noise_dim: int = 32,
        hidden_dim: int = 1024,
        feature_dim: int = 512,
        noise_bound: float = 1.5,
        resnet_weights: str = "imagenet",
    ) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(
            obs_state_dim=obs_state_dim,
            feature_dim=feature_dim,
            resnet_weights=resnet_weights,
        )
        self.actor = LatentActor(feature_dim, latent_noise_dim, hidden_dim, noise_bound)
        self.critic1 = Critic(feature_dim, latent_noise_dim, hidden_dim)
        self.critic2 = Critic(feature_dim, latent_noise_dim, hidden_dim)
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        self.log_alpha = nn.Parameter(torch.zeros(1))

    def encode_obs(self, batch: dict[str, Tensor], next_obs: bool = False) -> Tensor:
        prefix = "next_" if next_obs else ""
        return self.encoder(
            batch[f"{prefix}obs_state"],
            batch[f"{prefix}image_main"],
            batch[f"{prefix}image_wrist"],
        )

    @torch.no_grad()
    def update_targets(self, tau: float) -> None:
        for target, source in (
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.lerp_(source_param.data, tau)

