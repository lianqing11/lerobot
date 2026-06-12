#!/usr/bin/env python

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DSRLDatasetSummary:
    episodes: int
    sidecar_rows: int
    transitions: int
    skipped_unexecuted: int
    reward_values: tuple[int, ...]
    obs_state_dim: int
    noise_shape: tuple[int, int]
    repeat_noise_max_diff: float


class Pi05SidecarDataset(Dataset):
    """Chunk-level transitions for Pi0.5 latent-noise SAC.

    Each sample is one decision transition:
      observation_t -> latent_noise_t -> reward_t, observation_{t+1}, done

    The SAC-trained variable is `latent_noise`, not the robot action.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        image_size: int = 224,
        noise_reduction: str = "first",
        skip_unexecuted: bool = True,
    ) -> None:
        if noise_reduction != "first":
            raise ValueError("Only noise_reduction='first' is implemented for v1")

        self.root = Path(dataset_root)
        self.image_size = image_size
        self.noise_reduction = noise_reduction
        self.skip_unexecuted = skip_unexecuted
        self.transitions: list[tuple[dict, dict, bool]] = []
        self.summary = self._load()

    def _load(self) -> DSRLDatasetSummary:
        sidecar_paths = sorted((self.root / "inferences").glob("episode_*.parquet"))
        if not sidecar_paths:
            raise FileNotFoundError(f"No sidecar parquet files found under {self.root / 'inferences'}")

        episodes = 0
        sidecar_rows = 0
        skipped_unexecuted = 0
        reward_values: set[int] = set()
        obs_state_dim: int | None = None
        noise_shape: tuple[int, int] | None = None
        repeat_noise_max_diff = 0.0

        for path in sidecar_paths:
            table = pq.read_table(path)
            required_cols = {
                "obs_state",
                "noise",
                "noise_shape",
                "executed_count",
                "task_success",
                "obs_image_main_jpeg",
                "obs_image_wrist_jpeg",
            }
            missing = required_cols - set(table.column_names)
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

            rows = table.to_pylist()
            sidecar_rows += len(rows)
            valid_rows = []
            for row in rows:
                reward = int(row["task_success"])
                if reward < 0:
                    continue
                if self.skip_unexecuted and int(row["executed_count"]) == 0:
                    skipped_unexecuted += 1
                    continue

                current_obs_dim = len(row["obs_state"])
                if obs_state_dim is None:
                    obs_state_dim = current_obs_dim
                elif obs_state_dim != current_obs_dim:
                    raise ValueError(f"Inconsistent obs_state dim in {path}: {current_obs_dim} != {obs_state_dim}")

                current_noise_shape = tuple(int(x) for x in row["noise_shape"])
                if current_noise_shape != (50, 32):
                    raise ValueError(f"Expected noise_shape [50, 32], got {current_noise_shape} in {path}")
                noise_shape = current_noise_shape

                noise = np.asarray(row["noise"], dtype=np.float32).reshape(current_noise_shape)
                repeat_noise_max_diff = max(
                    repeat_noise_max_diff,
                    float(np.max(np.abs(noise - noise[0:1]))),
                )
                row["_latent_noise"] = noise[0].copy()
                row["_obs_state"] = np.asarray(row["obs_state"], dtype=np.float32)
                reward_values.add(reward)
                valid_rows.append(row)

            if len(valid_rows) >= 2:
                episodes += 1
                for i, row in enumerate(valid_rows):
                    is_last = i == len(valid_rows) - 1
                    next_row = row if is_last else valid_rows[i + 1]
                    self.transitions.append((row, next_row, is_last))

        if not self.transitions:
            raise ValueError(f"No valid transitions found in {self.root}")

        return DSRLDatasetSummary(
            episodes=episodes,
            sidecar_rows=sidecar_rows,
            transitions=len(self.transitions),
            skipped_unexecuted=skipped_unexecuted,
            reward_values=tuple(sorted(reward_values)),
            obs_state_dim=int(obs_state_dim or 0),
            noise_shape=noise_shape or (0, 0),
            repeat_noise_max_diff=repeat_noise_max_diff,
        )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row, next_row, is_last = self.transitions[idx]
        reward = float(int(row["task_success"])) if is_last else 0.0
        return {
            "obs_state": torch.from_numpy(row["_obs_state"]),
            "next_obs_state": torch.from_numpy(next_row["_obs_state"]),
            "image_main": self._decode_image(row["obs_image_main_jpeg"]),
            "image_wrist": self._decode_image(row["obs_image_wrist_jpeg"]),
            "next_image_main": self._decode_image(next_row["obs_image_main_jpeg"]),
            "next_image_wrist": self._decode_image(next_row["obs_image_wrist_jpeg"]),
            "latent_noise": torch.from_numpy(row["_latent_noise"]),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "done": torch.tensor([is_last], dtype=torch.float32),
        }

    def _decode_image(self, jpeg_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
        return (tensor - mean) / std


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def downsample_images(batch: dict[str, torch.Tensor], image_size: int) -> dict[str, torch.Tensor]:
    if batch["image_main"].shape[-1] == image_size:
        return batch
    for key in ("image_main", "image_wrist", "next_image_main", "next_image_wrist"):
        batch[key] = F.interpolate(batch[key], size=(image_size, image_size), mode="bilinear", align_corners=False)
    return batch

