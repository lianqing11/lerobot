#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from collections import deque

import pytest
import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import ActionEmbedder, PI05Policy, make_ic_all_frame_att_2d_masks
from lerobot.policies.pi05.processor_pi05 import (
    PI05_XVLA_ACTION_PAD,
    PI05_XVLA_FRAME_START,
    Pi05PrepareStateTokenizerProcessorStep,
    format_pi05_xvla_ic_prompt,
    next_obs_key,
)
from lerobot.processor.core import TransitionKey
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


def test_pi05_ic_delta_indices_and_sampler_drop():
    config = PI05Config(num_ic_frames=4, chunk_size=50, dtype="float32")

    assert config.ic_loss_mode == "all_frames"
    assert config.observation_delta_indices == [-150, -100, -50, 0]
    assert len(config.action_delta_indices) == 4 * config.chunk_size
    assert config.action_delta_indices[:3] == [-150, -149, -148]
    assert config.action_delta_indices[50:53] == [-100, -99, -98]
    assert config.action_delta_indices[-3:] == [47, 48, 49]
    assert config.drop_n_first_frames == 150
    assert config.tokenizer_max_length == 512


def test_pi05_ic_fixed_gap_delta_indices_include_explicit_next_obs():
    config = PI05Config(
        num_ic_frames=3,
        chunk_size=10,
        n_action_steps=10,
        ic_gap_range=[2, 2],
        dtype="float32",
    )

    assert config.observation_delta_indices == [-24, -12, 0, -14, -2]
    assert config.action_delta_indices[:3] == [-24, -23, -22]
    assert config.action_delta_indices[10:13] == [-12, -11, -10]
    assert config.action_delta_indices[-3:] == [7, 8, 9]
    assert config.drop_n_first_frames == 24


def test_pi05_ic_rejects_variable_gap_range():
    with pytest.raises(ValueError, match="fixed ic_gap_range"):
        PI05Config(num_ic_frames=3, ic_gap_range=[0, 2], dtype="float32")


def test_pi05_baseline_delta_indices_unchanged():
    config = PI05Config(chunk_size=50, dtype="float32")

    assert config.observation_delta_indices is None
    assert config.action_delta_indices == list(range(50))
    assert config.drop_n_first_frames == 0
    assert config.tokenizer_max_length == 200


def test_pi05_ic_processor_formats_context_state_prompt():
    step = Pi05PrepareStateTokenizerProcessorStep(max_state_dim=2, include_next_obs=True)
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.tensor(
                [
                    [
                        [-1.0, -0.5],
                        [0.0, 0.5],
                        [0.8, 1.0],
                    ]
                ],
                dtype=torch.float32,
            )
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_object"]},
    }

    output = step(transition)
    prompt = output[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt.startswith("Task: pick object;")
    assert "Demo 1 State:" in prompt
    assert "Next State:" in prompt
    assert "Demo 2 State:" in prompt
    assert "Query State:" in prompt
    assert prompt.endswith("Action: ")
    assert next_obs_key(OBS_STATE) in output[TransitionKey.OBSERVATION]
    assert output[TransitionKey.OBSERVATION][next_obs_key(OBS_STATE)].shape == (1, 2, 2)


def test_pi05_ic_processor_splits_explicit_next_observation_frames():
    step = Pi05PrepareStateTokenizerProcessorStep(
        max_state_dim=2,
        include_next_obs=True,
        num_ic_frames=3,
    )
    image_key = f"{OBS_IMAGES}.cam"
    states = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    images = torch.arange(1 * 5 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 5, 3, 2, 2)
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: states.clone(),
            image_key: images.clone(),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["pick_object"]},
    }

    output = step(transition)
    observation = output[TransitionKey.OBSERVATION]

    assert torch.equal(observation[OBS_STATE], states[:, :3])
    assert torch.equal(observation[next_obs_key(OBS_STATE)], states[:, 3:])
    assert torch.equal(observation[image_key], images[:, :3])
    assert torch.equal(observation[next_obs_key(image_key)], images[:, 3:])


def test_pi05_ic_processor_formats_xvla_prompt_and_next_images():
    step = Pi05PrepareStateTokenizerProcessorStep(
        max_state_dim=2,
        include_next_obs=True,
        ic_sequence_mode="xvla",
        num_action_tokens=3,
        use_frame_sep=True,
    )
    image_key = f"{OBS_IMAGES}.cam"
    transition = {
        TransitionKey.OBSERVATION: {
            OBS_STATE: torch.zeros(1, 3, 2),
            image_key: torch.zeros(1, 3, 3, 4, 4),
        },
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["open_drawer"]},
    }

    output = step(transition)
    prompt = output[TransitionKey.COMPLEMENTARY_DATA]["task"][0]

    assert prompt.count(PI05_XVLA_FRAME_START) == 3
    assert prompt.count(PI05_XVLA_ACTION_PAD) == 6
    assert "Query State:" in prompt
    assert next_obs_key(OBS_STATE) in output[TransitionKey.OBSERVATION]
    assert next_obs_key(image_key) in output[TransitionKey.OBSERVATION]
    assert output[TransitionKey.OBSERVATION][next_obs_key(image_key)].shape == (1, 2, 3, 4, 4)


def test_pi05_xvla_prompt_action_pad_count():
    prompt = format_pi05_xvla_ic_prompt(
        "pick",
        torch.zeros(4, 2).numpy(),
        num_action_tokens=5,
        include_next_obs=False,
    )

    assert prompt.count(PI05_XVLA_FRAME_START) == 4
    assert prompt.count(PI05_XVLA_ACTION_PAD) == 15


def test_action_embedder_shape():
    embedder = ActionEmbedder(dim_action=8, num_actions=5, output_dim=16, num_tokens=4)
    actions = torch.randn(2, 3, 5, 8)

    action_tokens = embedder(actions)

    assert action_tokens.shape == (2, 3, 4, 16)


def test_pi05_ic_inference_extracts_raw_task_from_existing_prompt():
    policy = object.__new__(PI05Policy)

    task = policy._extract_task_text("Task: pick object, State: 1 2 3;\nAction: ")

    assert task == "pick object"


def test_pi05_ic_peft_saves_action_embedder():
    policy = object.__new__(PI05Policy)
    policy.config = PI05Config(num_ic_frames=4, dtype="float32")

    targets = policy._get_default_peft_targets()

    assert targets["modules_to_save"] == ["model.action_embedder"]


def test_pi05_ic_peft_saves_frame_position_embedding_when_enabled():
    policy = object.__new__(PI05Policy)
    policy.config = PI05Config(
        num_ic_frames=4,
        ic_sequence_mode="xvla",
        use_frame_position_embed=True,
        dtype="float32",
    )

    targets = policy._get_default_peft_targets()

    assert targets["modules_to_save"] == ["model.action_embedder", "model.frame_position_embed"]


def test_pi05_all_frame_config_and_weights():
    config = PI05Config(
        num_ic_frames=4,
        chunk_size=50,
        ic_loss_mode="all_frames",
        ic_frame_loss_weights=[1.0, 1.0, 2.0, 2.0],
        dtype="float32",
    )

    assert config.ic_loss_mode == "all_frames"
    assert config.ic_frame_loss_weights == [1.0, 1.0, 2.0, 2.0]


def test_pi05_all_frame_attention_mask_is_frame_causal():
    prefix_pad = torch.ones(1, 3, dtype=torch.bool)
    suffix_pad = torch.ones(1, 6, dtype=torch.bool)
    prefix_frames = torch.tensor([0, 1, 2])
    suffix_frames = torch.tensor([0, 0, 1, 1, 2, 2])

    mask = make_ic_all_frame_att_2d_masks(prefix_pad, prefix_frames, suffix_pad, suffix_frames)[0]
    prefix_len = prefix_pad.shape[1]

    assert not mask[0, prefix_len]
    assert not mask[prefix_len, 1]
    assert mask[prefix_len + 2, 1]
    assert not mask[prefix_len + 2, 2]
    assert mask[prefix_len + 4, 2]
    assert mask[prefix_len, prefix_len + 1]
    assert not mask[prefix_len, prefix_len + 2]
    assert not mask[prefix_len + 2, prefix_len]


def test_pi05_all_frame_attention_mask_accepts_batched_prefix_frames():
    prefix_pad = torch.ones(2, 3, dtype=torch.bool)
    suffix_pad = torch.ones(2, 4, dtype=torch.bool)
    prefix_frames = torch.tensor([[0, 1, 2], [0, 0, 1]])
    suffix_frames = torch.tensor([0, 0, 1, 1])

    mask = make_ic_all_frame_att_2d_masks(prefix_pad, prefix_frames, suffix_pad, suffix_frames)

    assert mask.shape == (2, 7, 7)
    assert not mask[0, 3, 1]
    assert mask[1, 5, 1]
    assert not mask[1, 3, 2]


def test_pi05_ic_history_selection_modes():
    policy = object.__new__(PI05Policy)
    policy.config = PI05Config(num_ic_frames=3, dtype="float32")
    policy._ic_history = deque([{"id": i} for i in range(5)], maxlen=5)

    policy.config.ic_demo_mode = "normal"
    policy.config.ic_stride = 2
    assert [entry["id"] for entry in policy._select_ic_history_entries()] == [2, 4]

    policy.config.ic_demo_mode = "early"
    policy.config.ic_stride = 2
    assert [entry["id"] for entry in policy._select_ic_history_entries()] == [0, 2]

    policy.config.ic_demo_mode = "fixed_first"
    assert [entry["id"] for entry in policy._select_ic_history_entries()] == [0, 0]
