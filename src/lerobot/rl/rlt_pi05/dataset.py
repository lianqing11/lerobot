#!/usr/bin/env python

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RLTDatasetSummary:
    datasets: int
    episodes: int
    sidecar_rows: int
    transitions: int
    skipped_unlabeled: int
    skipped_unexecuted: int
    reward_values: tuple[int, ...]
    obs_state_dim: int
    action_dim: int
    chunk_C: int


def resolve_dataset_dirs(
    dataset_root: str | Path,
    includes: list[str] | None = None,
    dataset_list: str | Path | None = None,
) -> list[Path]:
    root = Path(dataset_root)
    if dataset_list is not None:
        dataset_dirs = []
        for entry in read_dataset_list(dataset_list):
            path = Path(entry)
            candidates = [path] if path.is_absolute() else sorted(root.glob(entry))
            if not candidates and not path.is_absolute():
                candidates = [root / entry]
            for candidate in candidates:
                if (candidate / "inferences").is_dir():
                    dataset_dirs.append(candidate)
                else:
                    raise FileNotFoundError(f"Dataset list entry does not contain inferences/: {candidate}")
        unique_dirs = sorted(set(dataset_dirs))
        if not unique_dirs:
            raise FileNotFoundError(f"No dataset directories found from list {dataset_list}")
        return unique_dirs

    if (root / "inferences").is_dir():
        return [root]

    patterns = includes or ["*"]
    dataset_dirs: list[Path] = []
    for pattern in patterns:
        dataset_dirs.extend(path for path in sorted(root.glob(pattern)) if (path / "inferences").is_dir())

    unique_dirs = sorted(set(dataset_dirs))
    if not unique_dirs:
        raise FileNotFoundError(f"No dataset directories with inferences/ found under {root}")
    return unique_dirs


def read_dataset_list(dataset_list: str | Path) -> list[str]:
    entries = []
    for line in Path(dataset_list).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


class RLTSidecarDataset(Dataset):
    """Chunk-level transitions for RLT action-prefix actor-critic training."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        includes: list[str] | None = None,
        dataset_list: str | Path | None = None,
        chunk_C: int = 16,
        action_dim: int = 7,
        image_size: int = 128,
        skip_unexecuted: bool = True,
    ) -> None:
        if not 1 <= chunk_C <= 50:
            raise ValueError(f"chunk_C must be in [1, 50], got {chunk_C}")

        self.dataset_dirs = resolve_dataset_dirs(dataset_root, includes, dataset_list)
        self.chunk_C = chunk_C
        self.action_dim = action_dim
        self.image_size = image_size
        self.skip_unexecuted = skip_unexecuted
        self.transitions: list[tuple[dict, dict, bool]] = []
        self.summary = self._load()

    def _load(self) -> RLTDatasetSummary:
        episodes = 0
        sidecar_rows = 0
        skipped_unlabeled = 0
        skipped_unexecuted = 0
        reward_values: set[int] = set()
        obs_state_dim: int | None = None

        for dataset_dir in self.dataset_dirs:
            for path in sorted((dataset_dir / "inferences").glob("episode_*.parquet")):
                table = pq.read_table(path)
                required_cols = {
                    "chunk_id",
                    "obs_state",
                    "obs_image_main_jpeg",
                    "obs_image_wrist_jpeg",
                    "action_chunk_proc",
                    "action_chunk_shape",
                    "vla_reference_chunk",
                    "task_success",
                    "executed_count",
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
                        skipped_unlabeled += 1
                        continue
                    if self.skip_unexecuted and int(row["executed_count"]) == 0:
                        skipped_unexecuted += 1
                        continue

                    current_obs_dim = len(row["obs_state"])
                    if obs_state_dim is None:
                        obs_state_dim = current_obs_dim
                    elif obs_state_dim != current_obs_dim:
                        raise ValueError(f"Inconsistent obs_state dim in {path}: {current_obs_dim} != {obs_state_dim}")

                    action_shape = tuple(int(x) for x in row["action_chunk_shape"])
                    if action_shape != (50, self.action_dim):
                        raise ValueError(f"Expected action_chunk_shape [50, {self.action_dim}], got {action_shape} in {path}")

                    row["_obs_state"] = np.asarray(row["obs_state"], dtype=np.float32)
                    row["_ref"] = self._chunk_prefix(row["vla_reference_chunk"], "vla_reference_chunk", path)
                    row["_action"] = self._chunk_prefix(row["action_chunk_proc"], "action_chunk_proc", path)
                    reward_values.add(reward)
                    valid_rows.append(row)

                valid_rows.sort(key=lambda item: int(item["chunk_id"]))
                if valid_rows:
                    episodes += 1
                    for i, row in enumerate(valid_rows):
                        is_last = i == len(valid_rows) - 1
                        next_row = row if is_last else valid_rows[i + 1]
                        self.transitions.append((row, next_row, is_last))

        if not self.transitions:
            raise ValueError(f"No valid RLT transitions found in {self.dataset_dirs}")

        return RLTDatasetSummary(
            datasets=len(self.dataset_dirs),
            episodes=episodes,
            sidecar_rows=sidecar_rows,
            transitions=len(self.transitions),
            skipped_unlabeled=skipped_unlabeled,
            skipped_unexecuted=skipped_unexecuted,
            reward_values=tuple(sorted(reward_values)),
            obs_state_dim=int(obs_state_dim or 0),
            action_dim=self.action_dim,
            chunk_C=self.chunk_C,
        )

    def _chunk_prefix(self, values: list[float], name: str, path: Path) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        expected = 50 * self.action_dim
        if arr.size != expected:
            raise ValueError(f"Expected {name} to have {expected} values, got {arr.size} in {path}")
        return arr.reshape(50, self.action_dim)[: self.chunk_C].copy()

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
            "ref": torch.from_numpy(row["_ref"]),
            "next_ref": torch.from_numpy(next_row["_ref"]),
            "action": torch.from_numpy(row["_action"]),
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
