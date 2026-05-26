#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

NEXT_OBS_PREFIX = "next."
PI05_PALIGEMMA_BASE_VOCAB_SIZE = 257152
PI05_XVLA_FRAME_START = "<|frame_start|>"
PI05_XVLA_FRAME_END = "<|frame_end|>"
PI05_XVLA_FRAME_SEP = "<|frame_sep|>"
PI05_XVLA_ACTION_START = "<|action_start|>"
PI05_XVLA_ACTION_PAD = "<|action_pad|>"
PI05_XVLA_ACTION_END = "<|action_end|>"
PI05_XVLA_SPECIAL_TOKENS = [
    PI05_XVLA_FRAME_START,
    PI05_XVLA_FRAME_END,
    PI05_XVLA_FRAME_SEP,
    PI05_XVLA_ACTION_START,
    PI05_XVLA_ACTION_PAD,
    PI05_XVLA_ACTION_END,
]


def _clean_task_text(task: str) -> str:
    return task.strip().replace("_", " ").replace("\n", " ")


def _discretize_normalized_state(state: torch.Tensor) -> np.ndarray:
    state_np = state.detach().cpu().numpy()
    return np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1


def _format_state_values(state: np.ndarray) -> str:
    return " ".join(map(str, state.tolist()))


def format_pi05_prompt(task: str, discretized_state: np.ndarray) -> str:
    state_str = _format_state_values(discretized_state)
    return f"Task: {_clean_task_text(task)}, State: {state_str};\nAction: "


def format_pi05_ic_prompt(
    task: str,
    discretized_states: np.ndarray,
    include_next_obs: bool = True,
    next_discretized_states: np.ndarray | None = None,
) -> str:
    parts = [f"Task: {_clean_task_text(task)};"]
    num_frames = discretized_states.shape[0]
    for frame_idx in range(num_frames - 1):
        state_str = _format_state_values(discretized_states[frame_idx])
        if include_next_obs:
            next_state = (
                next_discretized_states[frame_idx]
                if next_discretized_states is not None
                else discretized_states[frame_idx + 1]
            )
            next_state_str = _format_state_values(next_state)
            parts.append(f"Demo {frame_idx + 1} State: {state_str}; Next State: {next_state_str};")
        else:
            parts.append(f"Demo {frame_idx + 1} State: {state_str};")
    query_state_str = _format_state_values(discretized_states[-1])
    parts.append(f"Query State: {query_state_str};\nAction: ")
    return " ".join(parts)


def format_pi05_xvla_ic_prompt(
    task: str,
    discretized_states: np.ndarray,
    *,
    num_action_tokens: int,
    include_next_obs: bool = True,
    use_frame_sep: bool = False,
    next_discretized_states: np.ndarray | None = None,
) -> str:
    action_pads = " ".join([PI05_XVLA_ACTION_PAD] * num_action_tokens)
    parts = [f"Task: {_clean_task_text(task)};"]
    num_frames = discretized_states.shape[0]
    frame_parts = []
    for frame_idx in range(num_frames):
        state_str = _format_state_values(discretized_states[frame_idx])
        if frame_idx < num_frames - 1:
            frame = (
                f"{PI05_XVLA_FRAME_START} State: {state_str}; "
                f"{PI05_XVLA_ACTION_START} {action_pads} {PI05_XVLA_ACTION_END}"
            )
            if include_next_obs:
                next_state = (
                    next_discretized_states[frame_idx]
                    if next_discretized_states is not None
                    else discretized_states[frame_idx + 1]
                )
                next_state_str = _format_state_values(next_state)
                frame += f" Next State: {next_state_str};"
            frame += f" {PI05_XVLA_FRAME_END}"
        else:
            frame = f"{PI05_XVLA_FRAME_START} Query State: {state_str}; {PI05_XVLA_FRAME_END}"
        frame_parts.append(frame)

    joiner = f" {PI05_XVLA_FRAME_SEP} " if use_frame_sep else " "
    parts.append(joiner.join(frame_parts))
    parts.append("Action: ")
    return " ".join(parts)


def next_obs_key(key: str) -> str:
    return f"{NEXT_OBS_PREFIX}{key}"


@ProcessorStepRegistry.register(name="pi05_prepare_state_tokenizer_processor_step")
@dataclass
class Pi05PrepareStateTokenizerProcessorStep(ProcessorStep):
    """
    Processor step to prepare the state and tokenize the language input.
    """

    max_state_dim: int = 32
    task_key: str = "task"
    include_next_obs: bool = True
    ic_sequence_mode: str = "legacy"
    num_ic_frames: int = 1
    num_action_tokens: int = 4
    use_frame_sep: bool = False

    def _split_ic_tensor(self, key: str, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_frames = self.num_ic_frames if self.num_ic_frames > 1 else value.shape[1]
        if not self.include_next_obs or num_frames <= 1:
            return value, None
        if value.shape[1] == num_frames:
            return value, value[:, 1:]
        explicit_frames = num_frames + num_frames - 1
        if value.shape[1] == explicit_frames:
            return value[:, :num_frames], value[:, num_frames:]
        raise ValueError(
            f"PI05 IC feature {key} must have {num_frames} frame observations "
            f"or {explicit_frames} observations including explicit next.* frames, got {value.shape[1]}"
        )

    def _prepare_ic_observation_fields(
        self, transition: EnvTransition, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        observation = transition[TransitionKey.OBSERVATION]
        state, next_state = self._split_ic_tensor(OBS_STATE, state)
        observation[OBS_STATE] = state
        if next_state is not None:
            observation[next_obs_key(OBS_STATE)] = next_state

        for key, value in list(observation.items()):
            if not key.startswith(OBS_IMAGES) or key.startswith(NEXT_OBS_PREFIX):
                continue
            if isinstance(value, torch.Tensor) and value.ndim >= 5:
                value, next_value = self._split_ic_tensor(key, value)
                observation[key] = value
                if next_value is not None:
                    observation[next_obs_key(key)] = next_value
        return state, next_state

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()

        state = transition.get(TransitionKey.OBSERVATION, {}).get(OBS_STATE)
        if state is None:
            raise ValueError("State is required for PI05")
        tasks = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(self.task_key)
        if tasks is None:
            raise ValueError("No task found in complementary data")

        # TODO: check if this necessary
        state = deepcopy(state)

        full_prompts = []
        if state.ndim == 3:
            state, next_state = self._prepare_ic_observation_fields(transition, state)
            discretized_states = _discretize_normalized_state(state)
            next_discretized_states = (
                _discretize_normalized_state(next_state) if next_state is not None else None
            )
            for i, task in enumerate(tasks):
                sample_next_states = (
                    next_discretized_states[i] if next_discretized_states is not None else None
                )
                if self.ic_sequence_mode == "xvla":
                    full_prompts.append(
                        format_pi05_xvla_ic_prompt(
                            task,
                            discretized_states[i],
                            num_action_tokens=self.num_action_tokens,
                            include_next_obs=self.include_next_obs,
                            use_frame_sep=self.use_frame_sep,
                            next_discretized_states=sample_next_states,
                        )
                    )
                else:
                    full_prompts.append(
                        format_pi05_ic_prompt(
                            task,
                            discretized_states[i],
                            include_next_obs=self.include_next_obs,
                            next_discretized_states=sample_next_states,
                        )
                    )
        else:
            discretized_states = _discretize_normalized_state(state)
            for i, task in enumerate(tasks):
                full_prompts.append(format_pi05_prompt(task, discretized_states[i]))

        transition[TransitionKey.COMPLEMENTARY_DATA][self.task_key] = full_prompts
        # Normalize state to [-1, 1] range if needed (assuming it's already normalized by normalizer processor step!!)
        # Discretize into 256 bins (see openpi `PaligemmaTokenizer.tokenize()`)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        This step does not alter the feature definitions.
        """
        return features


def make_pi05_pre_post_processors(
    config: PI05Config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Constructs pre-processor and post-processor pipelines for the PI0 policy.

    The pre-processing pipeline prepares input data for the model by:
    1. Renaming features to match pretrained configurations.
    2. Normalizing input and output features based on dataset statistics.
    3. Adding a batch dimension.
    4. Appending a newline character to the task description for tokenizer compatibility.
    5. Tokenizing the text prompt using the PaliGemma tokenizer.
    6. Moving all data to the specified device.

    The post-processing pipeline handles the model's output by:
    1. Moving data to the CPU.
    2. Unnormalizing the output features to their original scale.

    Args:
        config: The configuration object for the PI0 policy.
        dataset_stats: A dictionary of statistics for normalization.
        preprocessor_kwargs: Additional arguments for the pre-processor pipeline.
        postprocessor_kwargs: Additional arguments for the post-processor pipeline.

    Returns:
        A tuple containing the configured pre-processor and post-processor pipelines.
    """

    # Add remaining processors
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),  # To mimic the same processor as pretrained one
        AddBatchDimensionProcessorStep(),
        # NOTE: NormalizerProcessorStep MUST come before Pi05PrepareStateTokenizerProcessorStep
        # because the tokenizer step expects normalized state in [-1, 1] range for discretization
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        Pi05PrepareStateTokenizerProcessorStep(
            max_state_dim=config.max_state_dim,
            include_next_obs=config.include_next_obs,
            ic_sequence_mode=config.ic_sequence_mode,
            num_ic_frames=config.num_ic_frames,
            num_action_tokens=config.num_action_tokens,
            use_frame_sep=config.use_frame_sep,
        ),
        TokenizerProcessorStep(
            tokenizer_name="google/paligemma-3b-pt-224",
            max_length=config.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
            special_tokens=PI05_XVLA_SPECIAL_TOKENS if config.ic_sequence_mode == "xvla" else None,
        ),
        DeviceProcessorStep(device=config.device),
    ]

    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
