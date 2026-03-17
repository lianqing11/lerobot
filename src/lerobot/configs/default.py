#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.datasets.transforms import ImageTransformsConfig
from lerobot.datasets.video_utils import get_safe_default_codec


def _parse_dataset_list_file(filepath: str) -> tuple[list[str], list[str]]:
    """Parse a text file listing dataset paths (one per line).

    Returns (repo_ids, roots) where repo_id is derived from the directory basename.
    Lines starting with '#' and empty lines are ignored.
    Each line can be:
      - A dataset root path: ``/data/pick_cup``  -> repo_id="pick_cup", root="/data/pick_cup"
      - repo_id and root separated by whitespace: ``pick_cup /data/pick_cup``
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"dataset_list_file not found: {filepath}")

    repo_ids: list[str] = []
    roots: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            root_path = parts[0].rstrip("/")
            repo_id = Path(root_path).name
            roots.append(root_path)
            repo_ids.append(repo_id)
        elif len(parts) == 2:
            repo_ids.append(parts[0])
            roots.append(parts[1].rstrip("/"))
        else:
            raise ValueError(
                f"Invalid line in dataset list file: '{raw_line}'. "
                "Expected either a single path or 'repo_id path'."
            )

    if not repo_ids:
        raise ValueError(f"No datasets found in {filepath}")
    return repo_ids, roots


@dataclass
class DatasetConfig:
    # You may provide a single repo_id (str) or a list of repo_ids (list[str]).
    # When a list is provided, train.py creates all datasets and concatenates them via
    # MultiLeRobotDataset. Only data keys common across all datasets are kept. Each dataset
    # gets an additional "dataset_index" field in returned items.
    repo_id: str | list[str] = ""
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    # For multi-dataset: provide a list of roots matching repo_id order, or a single
    # shared parent directory. If None, defaults to $HF_LEROBOT_HOME/repo_id.
    root: str | list[str] | None = None
    # Path to a text file listing multiple datasets (one per line).
    # Each line is either a dataset root path, or "repo_id root_path" separated by whitespace.
    # Lines starting with '#' and blank lines are ignored.
    # When set, this overrides repo_id and root with the parsed values.
    dataset_list_file: str | None = None
    # For single dataset: list[int] of episode indices.
    # For multi-dataset: dict mapping repo_id -> list[int], or None to use all episodes.
    episodes: list[int] | dict[str, list[int]] | None = None
    image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)
    revision: str | None = None
    use_imagenet_stats: bool = True
    video_backend: str = field(default_factory=get_safe_default_codec)
    streaming: bool = False

    def __post_init__(self) -> None:
        if self.dataset_list_file is not None:
            repo_ids, roots = _parse_dataset_list_file(self.dataset_list_file)
            if len(repo_ids) == 1:
                self.repo_id = repo_ids[0]
                self.root = roots[0]
            else:
                self.repo_id = repo_ids
                self.root = roots
        if not self.repo_id:
            raise ValueError(
                "No dataset specified. Provide either 'repo_id' or 'dataset_list_file'."
            )


@dataclass
class WandBConfig:
    enable: bool = False
    # Set to true to disable saving an artifact despite training.save_checkpoint=True
    disable_artifact: bool = False
    project: str = "lerobot"
    entity: str | None = None
    notes: str | None = None
    run_id: str | None = None
    mode: str | None = None  # Allowed values: 'online', 'offline' 'disabled'. Defaults to 'online'


@dataclass
class EvalConfig:
    n_episodes: int = 50
    # `batch_size` specifies the number of environments to use in a gym.vector.VectorEnv.
    batch_size: int = 50
    # `use_async_envs` specifies whether to use asynchronous environments (multiprocessing).
    use_async_envs: bool = False

    def __post_init__(self) -> None:
        if self.batch_size > self.n_episodes:
            raise ValueError(
                "The eval batch size is greater than the number of eval episodes "
                f"({self.batch_size} > {self.n_episodes}). As a result, {self.batch_size} "
                f"eval environments will be instantiated, but only {self.n_episodes} will be used. "
                "This might significantly slow down evaluation. To fix this, you should update your command "
                f"to increase the number of episodes to match the batch size (e.g. `eval.n_episodes={self.batch_size}`), "
                f"or lower the batch size (e.g. `eval.batch_size={self.n_episodes}`)."
            )


@dataclass
class PeftConfig:
    # PEFT offers many fine-tuning methods, layer adapters being the most common and currently also the most
    # effective methods so we'll focus on those in this high-level config interface.

    # Either a string (module name suffix or 'all-linear'), a list of module name suffixes or a regular expression
    # describing module names to target with the configured PEFT method. Some policies have a default value for this
    # so that you don't *have* to choose which layers to adapt but it might still be worthwhile depending on your case.
    target_modules: list[str] | str | None = None

    # Names/suffixes of modules to fully fine-tune and store alongside adapter weights. Useful for layers that are
    # not part of a pre-trained model (e.g., action state projections). Depending on the policy this defaults to layers
    # that are newly created in pre-trained policies. If you're fine-tuning an already trained policy you might want
    # to set this to `[]`. Corresponds to PEFT's `modules_to_save`.
    full_training_modules: list[str] | None = None

    # The PEFT (adapter) method to apply to the policy. Needs to be a valid PEFT type.
    method_type: str = "LORA"

    # Adapter initialization method. Look at the specific PEFT adapter documentation for defaults.
    init_type: str | None = None

    # We expect that all PEFT adapters are in some way doing rank-decomposition therefore this parameter specifies
    # the rank used for the adapter. In general a higher rank means more trainable parameters and closer to full
    # fine-tuning.
    r: int = 16
