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

import builtins
import copy
import logging
import math
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.import_utils import _transformers_available

# Conditional import for type checking and lazy loading
if TYPE_CHECKING or _transformers_available:
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.gemma import modeling_gemma

    from lerobot.policies.pi_gemma import (
        PaliGemmaForConditionalGenerationWithPiGemma,
        PiGemmaForCausalLM,
        _gated_residual,
        layernorm_forward,
    )
else:
    CONFIG_MAPPING = None
    modeling_gemma = None
    PiGemmaForCausalLM = None
    _gated_residual = None
    layernorm_forward = None
    PaliGemmaForConditionalGenerationWithPiGemma = None
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    PI05_PALIGEMMA_BASE_VOCAB_SIZE,
    PI05_XVLA_ACTION_END,
    PI05_XVLA_ACTION_PAD,
    PI05_XVLA_ACTION_START,
    PI05_XVLA_FRAME_END,
    PI05_XVLA_FRAME_SEP,
    PI05_XVLA_FRAME_START,
    PI05_XVLA_SPECIAL_TOKENS,
    _discretize_normalized_state,
    format_pi05_ic_prompt,
    format_pi05_xvla_ic_prompt,
    next_obs_key,
)
from lerobot.policies.pretrained import PreTrainedPolicy, T
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import (
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    OPENPI_ATTENTION_MASK_VALUE,
)


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    # Beta sampling uses _sample_dirichlet which isn't implemented for MPS, so sample on CPU
    alpha_t = torch.tensor(alpha, dtype=torch.float32)
    beta_t = torch.tensor(beta, dtype=torch.float32)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def make_ic_all_frame_att_2d_masks(
    prefix_pad_masks: Tensor,
    prefix_frame_ids: Tensor,
    suffix_pad_masks: Tensor,
    suffix_frame_ids: Tensor,
) -> Tensor:
    bsize = prefix_pad_masks.shape[0]
    prefix_len = prefix_pad_masks.shape[1]
    suffix_len = suffix_pad_masks.shape[1]

    if prefix_frame_ids.ndim == 1:
        prefix_frame_ids = prefix_frame_ids[None, :].expand(bsize, -1)
    if suffix_frame_ids.ndim == 1:
        suffix_frame_ids = suffix_frame_ids[None, :].expand(bsize, -1)

    prefix_query_frames = prefix_frame_ids[:, :, None]
    prefix_key_frames = prefix_frame_ids[:, None, :]
    suffix_query_frames = suffix_frame_ids[:, :, None]
    suffix_key_frames = suffix_frame_ids[:, None, :]

    prefix_prefix = prefix_key_frames <= prefix_query_frames
    prefix_suffix = torch.zeros(bsize, prefix_len, suffix_len, dtype=torch.bool, device=prefix_pad_masks.device)
    suffix_prefix = prefix_key_frames <= suffix_query_frames
    suffix_suffix = suffix_key_frames == suffix_query_frames

    top = torch.cat([prefix_prefix, prefix_suffix], dim=2)
    bottom = torch.cat([suffix_prefix, suffix_suffix], dim=2)
    att_2d_masks = torch.cat([top, bottom], dim=1)

    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


class ActionEmbedder(nn.Module):
    """Embed normalized action chunks as PaliGemma prefix tokens for IC context."""

    def __init__(
        self,
        dim_action: int,
        num_actions: int,
        output_dim: int,
        num_tokens: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim = output_dim
        hidden_dim = hidden_dim or output_dim
        self.net = nn.Sequential(
            nn.Linear(dim_action * num_actions, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_tokens * output_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        leading = actions.shape[:-2]
        action_tokens = self.net(actions.reshape(*leading, -1))
        return action_tokens.reshape(*leading, self.num_tokens, self.output_dim)


def resize_with_pad_torch(  # see openpi `resize_with_pad_torch` (exact copy)
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """PyTorch version of resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].

    Args:
        images: Tensor of shape [*b, h, w, c] or [*b, c, h, w]
        height: Target height
        width: Target width
        mode: Interpolation mode ('bilinear', 'nearest', etc.)

    Returns:
        Resized and padded tensor with same shape format as input
    """
    # Check if input is in channels-last format [*b, h, w, c] or channels-first [*b, c, h, w]
    if images.shape[-1] <= 4:  # Assume channels-last format
        channels_last = True
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension
        images = images.permute(0, 3, 1, 2)  # [b, h, w, c] -> [b, c, h, w]
    else:
        channels_last = False
        if images.dim() == 3:
            images = images.unsqueeze(0)  # Add batch dimension

    batch_size, channels, cur_height, cur_width = images.shape

    # Calculate resize ratio
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    # Resize
    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    # Handle dtype-specific clipping
    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    # Calculate padding
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w

    # Pad
    constant_value = 0 if images.dtype == torch.uint8 else 0.0
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),  # left, right, top, bottom
        mode="constant",
        value=constant_value,
    )

    # Convert back to original format if needed
    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

    return padded_images


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(
    layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond, paligemma, gemma_expert
):
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    # SDPA: PyTorch auto-dispatches to Flash Attention 2 / memory-efficient backend
    att_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
    )
    # (batch, heads, seq, head_dim) -> (batch, seq, heads * head_dim)
    num_heads = att_output.shape[1]
    att_output = att_output.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class GemmaConfig:  # see openpi `gemma.py: Config`
    """Configuration for Gemma model variants."""

    def __init__(self, width, depth, mlp_dim, num_heads, num_kv_heads, head_dim):
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


def get_gemma_config(variant: str) -> GemmaConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    if variant == "gemma_300m":
        return GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    elif variant == "gemma_2b":
        return GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class PaliGemmaWithExpertModel(
    nn.Module
):  # see openpi `gemma_pytorch.py: PaliGemmaWithExpertModel` this class is almost a exact copy of PaliGemmaWithExpertModel in openpi
    """PaliGemma model with action expert for PI05."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        freeze_vision_encoder: bool = False,
        train_expert_only: bool = False,
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.image_size = image_size
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.gemma_expert = PiGemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)
        self._set_requires_grad()

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        # Keep full vision path in float32 so we never toggle (toggle causes optimizer
        # "same dtype" error). Saves memory vs full float32; more memory than only 3 params.
        params_to_keep_float32 = [
            "vision_tower",
            "multi_modal_projector",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def _set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
            for param in self.paligemma.model.vision_tower.parameters():
                param.requires_grad = False
        if self.train_expert_only:
            self.paligemma.eval()
            for param in self.paligemma.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.paligemma.model.vision_tower.eval()
        if self.train_expert_only:
            self.paligemma.eval()

    def embed_image(self, image: torch.Tensor):
        # Vision tower and multi_modal_projector are kept in float32 (params_to_keep_float32).
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.paligemma.model.get_image_features(image)
        features = image_outputs.pooler_output * self.paligemma.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.model.language_model.embed_tokens(tokens)

    def resize_language_token_embeddings(self, vocab_size: int) -> None:
        old_embed = self.paligemma.model.language_model.embed_tokens
        if old_embed.num_embeddings >= vocab_size:
            return

        new_embed = nn.Embedding(
            vocab_size,
            old_embed.embedding_dim,
            padding_idx=old_embed.padding_idx,
            device=old_embed.weight.device,
            dtype=old_embed.weight.dtype,
        )
        with torch.no_grad():
            new_embed.weight[: old_embed.num_embeddings].copy_(old_embed.weight)
            std = old_embed.weight.float().std().item()
            nn.init.normal_(new_embed.weight[old_embed.num_embeddings :], mean=0.0, std=std)
        self.paligemma.model.language_model.embed_tokens = new_embed
        self.paligemma.config.text_config.vocab_size = vocab_size
        self.paligemma.config._vocab_size = vocab_size  # noqa: SLF001

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.model.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.model.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = layernorm_forward(models[i].norm, hidden_states, adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values


class PI05Pytorch(nn.Module):  # see openpi `PI0Pytorch`
    """Core PI05 PyTorch model."""

    def __init__(self, config: PI05Config, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config
        self.rtc_processor = rtc_processor

        paligemma_config = get_gemma_config(config.paligemma_variant)
        action_expert_config = get_gemma_config(config.action_expert_variant)

        if config.image_resolution[0] != config.image_resolution[1]:
            raise ValueError(
                f"PaliGemma expects square image resolution, invalid resolution: {config.image_resolution}"
            )

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True],
            precision=config.dtype,
            image_size=config.image_resolution[0],
            freeze_vision_encoder=config.freeze_vision_encoder,
            train_expert_only=config.train_expert_only,
        )
        if config.ic_sequence_mode == "xvla":
            self.paligemma_with_expert.resize_language_token_embeddings(
                PI05_PALIGEMMA_BASE_VOCAB_SIZE + len(PI05_XVLA_SPECIAL_TOKENS)
            )

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.max_action_dim)
        self.action_embedder = (
            ActionEmbedder(
                dim_action=config.max_action_dim,
                num_actions=config.chunk_size,
                output_dim=paligemma_config.width,
                num_tokens=config.num_action_tokens,
            )
            if config.num_ic_frames > 1
            else None
        )
        self.frame_position_embed = (
            nn.Embedding(config.num_ic_frames, paligemma_config.width)
            if config.ic_sequence_mode == "xvla" and config.use_frame_position_embed and config.num_ic_frames > 1
            else None
        )

        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for PI05Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.model.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.model.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for PI05Pytorch model")

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, tokens, masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def _append_image_embeds(self, images, img_masks, embs, pad_masks, att_masks) -> None:
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

    def embed_prefix_ic(
        self,
        images_by_frame,
        img_masks_by_frame,
        tokens,
        masks,
        context_actions,
        next_images_by_frame=None,
        next_img_masks_by_frame=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.action_embedder is None:
            raise ValueError("PI05 IC requires action_embedder; set num_ic_frames > 1")

        embs = []
        pad_masks = []
        att_masks = []

        num_frames = len(images_by_frame)
        num_context = num_frames - 1
        self._append_image_embeds(images_by_frame[0], img_masks_by_frame[0], embs, pad_masks, att_masks)

        embedder_dtype = next(self.action_embedder.parameters()).dtype
        if self.config.zero_ic_actions:
            context_actions = torch.zeros_like(context_actions)
        action_embeds = self.action_embedder(context_actions.to(dtype=embedder_dtype))
        for frame_idx in range(num_context):
            if frame_idx > 0:
                self._append_image_embeds(
                    images_by_frame[frame_idx], img_masks_by_frame[frame_idx], embs, pad_masks, att_masks
                )

            action_emb = action_embeds[:, frame_idx]
            embs.append(action_emb)
            pad_masks.append(torch.ones(action_emb.shape[:2], dtype=torch.bool, device=action_emb.device))
            att_masks += [0] * action_emb.shape[1]

            if self.config.include_next_obs:
                if next_images_by_frame is None or next_img_masks_by_frame is None:
                    raise ValueError("PI05 IC include_next_obs=True requires explicit next.* image fields")
                self._append_image_embeds(
                    next_images_by_frame[frame_idx],
                    next_img_masks_by_frame[frame_idx],
                    embs,
                    pad_masks,
                    att_masks,
                )

        if not self.config.include_next_obs:
            self._append_image_embeds(
                images_by_frame[-1], img_masks_by_frame[-1], embs, pad_masks, att_masks
            )

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        embs.append(lang_emb)
        pad_masks.append(masks)
        att_masks += [0] * lang_emb.shape[1]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(pad_masks.shape[0], len(att_masks))
        return embs, pad_masks, att_masks

    def _append_frame_ids(self, frame_ids, length: int, frame_idx: int) -> None:
        frame_ids.extend([frame_idx] * length)

    def _xvla_token_id(self, token: str) -> int:
        return PI05_PALIGEMMA_BASE_VOCAB_SIZE + PI05_XVLA_SPECIAL_TOKENS.index(token)

    def _parse_xvla_token_frame_ids(self, tokens: Tensor, masks: Tensor) -> Tensor:
        frame_start_id = self._xvla_token_id(PI05_XVLA_FRAME_START)
        frame_end_id = self._xvla_token_id(PI05_XVLA_FRAME_END)
        action_start_id = self._xvla_token_id(PI05_XVLA_ACTION_START)
        action_end_id = self._xvla_token_id(PI05_XVLA_ACTION_END)

        bsize, seq_len = tokens.shape
        frame_ids = torch.zeros(bsize, seq_len, dtype=torch.long, device=tokens.device)
        max_frame = self.config.num_ic_frames - 1
        for batch_idx in range(bsize):
            frame_idx = 0
            after_action = False
            for token_idx in range(seq_len):
                if not bool(masks[batch_idx, token_idx]):
                    continue
                token_id = int(tokens[batch_idx, token_idx].item())
                token_frame = min(frame_idx + int(after_action), max_frame)
                frame_ids[batch_idx, token_idx] = token_frame
                if token_id == frame_start_id:
                    after_action = False
                    frame_ids[batch_idx, token_idx] = frame_idx
                elif token_id == action_start_id:
                    after_action = True
                    frame_ids[batch_idx, token_idx] = min(frame_idx + 1, max_frame)
                elif token_id == action_end_id:
                    frame_ids[batch_idx, token_idx] = min(frame_idx + 1, max_frame)
                elif token_id == frame_end_id:
                    frame_ids[batch_idx, token_idx] = token_frame
                    frame_idx = min(frame_idx + 1, max_frame)
                    after_action = False
        return frame_ids

    def _scatter_xvla_action_embeds(self, lang_emb: Tensor, tokens: Tensor, masks: Tensor, actions: Tensor) -> Tensor:
        if self.action_embedder is None:
            raise ValueError("PI05 xvla IC requires action_embedder; set num_ic_frames > 1")

        context_actions = actions
        if self.config.zero_ic_actions:
            context_actions = torch.zeros_like(context_actions)
        embedder_dtype = next(self.action_embedder.parameters()).dtype
        action_embeds = self.action_embedder(context_actions.to(dtype=embedder_dtype)).to(dtype=lang_emb.dtype)
        flat_action_embeds = action_embeds.reshape(action_embeds.shape[0], -1, action_embeds.shape[-1])

        action_pad_id = self._xvla_token_id(PI05_XVLA_ACTION_PAD)
        action_pad_masks = (tokens == action_pad_id) & masks
        expected_tokens = flat_action_embeds.shape[1]
        lang_emb = lang_emb.clone()
        for batch_idx in range(tokens.shape[0]):
            positions = action_pad_masks[batch_idx].nonzero(as_tuple=True)[0]
            if positions.numel() != expected_tokens:
                raise ValueError(
                    f"PI05 xvla IC expected {expected_tokens} action pads, got {positions.numel()}"
                )
            lang_emb[batch_idx, positions] = flat_action_embeds[batch_idx]
        return lang_emb

    def _apply_frame_position_embeds(self, embs: Tensor, frame_ids: Tensor) -> Tensor:
        if self.frame_position_embed is None:
            return embs
        return embs + self.frame_position_embed(frame_ids.clamp(max=self.config.num_ic_frames - 1)).to(
            dtype=embs.dtype
        )

    def embed_prefix_ic_xvla(
        self,
        images_by_frame,
        img_masks_by_frame,
        tokens,
        masks,
        context_actions,
        next_images_by_frame=None,
        next_img_masks_by_frame=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.action_embedder is None:
            raise ValueError("PI05 xvla IC requires action_embedder; set num_ic_frames > 1")

        embs = []
        pad_masks = []
        frame_ids = []
        num_frames = len(images_by_frame)
        for frame_idx in range(num_frames):
            before_len = sum(emb.shape[1] for emb in embs)
            image_att_masks = []
            self._append_image_embeds(
                images_by_frame[frame_idx],
                img_masks_by_frame[frame_idx],
                embs,
                pad_masks,
                image_att_masks,
            )
            self._append_frame_ids(frame_ids, sum(emb.shape[1] for emb in embs) - before_len, frame_idx)
            if self.config.include_next_obs and frame_idx < num_frames - 1:
                if next_images_by_frame is None or next_img_masks_by_frame is None:
                    raise ValueError("PI05 xvla IC include_next_obs=True requires explicit next.* image fields")
                before_len = sum(emb.shape[1] for emb in embs)
                next_att_masks = []
                self._append_image_embeds(
                    next_images_by_frame[frame_idx],
                    next_img_masks_by_frame[frame_idx],
                    embs,
                    pad_masks,
                    next_att_masks,
                )
                self._append_frame_ids(
                    frame_ids, sum(emb.shape[1] for emb in embs) - before_len, frame_idx + 1
                )

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
        lang_emb = self._scatter_xvla_action_embeds(lang_emb, tokens, masks, context_actions)
        token_frame_ids = self._parse_xvla_token_frame_ids(tokens, masks)
        embs.append(lang_emb)
        pad_masks.append(masks)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        image_frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=pad_masks.device)
        image_frame_ids = image_frame_ids[None, :].expand(tokens.shape[0], -1)
        all_frame_ids = torch.cat([image_frame_ids, token_frame_ids], dim=1)
        embs = self._apply_frame_position_embeds(embs, all_frame_ids)
        return embs, pad_masks, all_frame_ids

    def embed_prefix_ic_all_frames(
        self,
        images_by_frame,
        img_masks_by_frame,
        tokens_by_frame,
        masks_by_frame,
        actions,
        next_images_by_frame=None,
        next_img_masks_by_frame=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.action_embedder is None:
            raise ValueError("PI05 all-frame IC requires action_embedder; set num_ic_frames > 1")

        embs = []
        pad_masks = []
        frame_ids = []
        num_frames = len(images_by_frame)

        context_actions = actions[:, :-1]
        if self.config.zero_ic_actions:
            context_actions = torch.zeros_like(context_actions)
        embedder_dtype = next(self.action_embedder.parameters()).dtype
        action_embeds = self.action_embedder(context_actions.to(dtype=embedder_dtype))

        def lang_embed_func(tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        for frame_idx in range(num_frames):
            before_len = sum(emb.shape[1] for emb in embs)
            image_att_masks = []
            self._append_image_embeds(
                images_by_frame[frame_idx],
                img_masks_by_frame[frame_idx],
                embs,
                pad_masks,
                image_att_masks,
            )
            self._append_frame_ids(frame_ids, sum(emb.shape[1] for emb in embs) - before_len, frame_idx)

            tokens = tokens_by_frame[:, frame_idx]
            masks = masks_by_frame[:, frame_idx]
            lang_emb = self._apply_checkpoint(lang_embed_func, tokens)
            embs.append(lang_emb)
            pad_masks.append(masks)
            self._append_frame_ids(frame_ids, lang_emb.shape[1], frame_idx)

            if frame_idx < num_frames - 1:
                action_emb = action_embeds[:, frame_idx]
                embs.append(action_emb)
                pad_masks.append(torch.ones(action_emb.shape[:2], dtype=torch.bool, device=action_emb.device))
                self._append_frame_ids(frame_ids, action_emb.shape[1], frame_idx + 1)
                if self.config.include_next_obs:
                    if next_images_by_frame is None or next_img_masks_by_frame is None:
                        raise ValueError("PI05 all-frame IC include_next_obs=True requires explicit next.* image fields")
                    before_len = sum(emb.shape[1] for emb in embs)
                    next_att_masks = []
                    self._append_image_embeds(
                        next_images_by_frame[frame_idx],
                        next_img_masks_by_frame[frame_idx],
                        embs,
                        pad_masks,
                        next_att_masks,
                    )
                    self._append_frame_ids(
                        frame_ids, sum(emb.shape[1] for emb in embs) - before_len, frame_idx + 1
                    )

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        frame_ids = torch.tensor(frame_ids, dtype=torch.long, device=pad_masks.device)
        return embs, pad_masks, frame_ids

    def embed_suffix_all_frames(self, noisy_actions, timestep):
        bsize, num_frames, chunk_size = noisy_actions.shape[:3]
        flat_actions = noisy_actions.reshape(bsize, num_frames * chunk_size, noisy_actions.shape[-1])

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(flat_actions):
            return self.action_in_proj(flat_actions)

        action_emb = self._apply_checkpoint(action_proj_func, flat_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        pad_masks = torch.ones(bsize, num_frames * chunk_size, dtype=torch.bool, device=noisy_actions.device)
        frame_ids = torch.arange(num_frames, device=noisy_actions.device).repeat_interleave(chunk_size)
        return action_emb, pad_masks, frame_ids, time_emb

    def embed_suffix(self, noisy_actions, timestep):
        """Embed noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_ic_query_only(
        self,
        images_by_frame,
        img_masks_by_frame,
        tokens,
        masks,
        context_actions,
        query_actions,
        next_images_by_frame=None,
        next_img_masks_by_frame=None,
        noise=None,
        time=None,
    ) -> Tensor:
        """IC training forward: context frames condition the query action loss only."""
        if self.config.ic_loss_mode != "query_only":
            raise NotImplementedError("PI05 all-frame IC loss is not implemented yet")

        if noise is None:
            noise = self.sample_noise(query_actions.shape, query_actions.device)

        if time is None:
            time = self.sample_time(query_actions.shape[0], query_actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * query_actions
        u_t = noise - query_actions

        if self.config.ic_sequence_mode == "xvla":
            prefix_embs, prefix_pad_masks, prefix_frame_ids = self.embed_prefix_ic_xvla(
                images_by_frame,
                img_masks_by_frame,
                tokens,
                masks,
                context_actions,
                next_images_by_frame,
                next_img_masks_by_frame,
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_ic(
                images_by_frame,
                img_masks_by_frame,
                tokens,
                masks,
                context_actions,
                next_images_by_frame,
                next_img_masks_by_frame,
            )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        if self.config.ic_sequence_mode == "xvla":
            suffix_frame_ids = torch.full(
                (suffix_pad_masks.shape[1],),
                len(images_by_frame) - 1,
                dtype=torch.long,
                device=suffix_pad_masks.device,
            )
            att_2d_masks = make_ic_all_frame_att_2d_masks(
                prefix_pad_masks,
                prefix_frame_ids,
                suffix_pad_masks,
                suffix_frame_ids,
            )
        else:
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    def forward_ic_all_frames(
        self,
        images_by_frame,
        img_masks_by_frame,
        tokens_by_frame,
        masks_by_frame,
        actions,
        next_images_by_frame=None,
        next_img_masks_by_frame=None,
        noise=None,
        time=None,
    ) -> Tensor:
        if self.config.ic_loss_mode != "all_frames":
            raise ValueError(f"forward_ic_all_frames requires ic_loss_mode='all_frames', got {self.config.ic_loss_mode}")

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        if self.config.ic_sequence_mode == "xvla":
            prefix_embs, prefix_pad_masks, prefix_frame_ids = self.embed_prefix_ic_xvla(
                images_by_frame,
                img_masks_by_frame,
                tokens_by_frame,
                masks_by_frame,
                actions[:, :-1],
                next_images_by_frame,
                next_img_masks_by_frame,
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_frame_ids = self.embed_prefix_ic_all_frames(
                images_by_frame,
                img_masks_by_frame,
                tokens_by_frame,
                masks_by_frame,
                actions,
                next_images_by_frame,
                next_img_masks_by_frame,
            )
        suffix_embs, suffix_pad_masks, suffix_frame_ids, adarms_cond = self.embed_suffix_all_frames(x_t, time)

        if (
            self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_2d_masks = make_ic_all_frame_att_2d_masks(
            prefix_pad_masks,
            prefix_frame_ids,
            suffix_pad_masks,
            suffix_frame_ids,
        )
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            att_2d_masks_4d,
            position_ids,
            adarms_cond,
        )
        suffix_out = suffix_out.reshape(
            actions.shape[0],
            actions.shape[1],
            actions.shape[2],
            suffix_out.shape[-1],
        )
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        tokens,
        masks,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, tokens, masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "sdpa"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=input_x_t,
                    timestep=current_timestep,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    @torch.no_grad()
    def sample_actions_ic(
        self,
        images_by_frame,
        img_masks_by_frame,
        tokens,
        masks,
        context_actions,
        next_images_by_frame=None,
        next_img_masks_by_frame=None,
        noise=None,
        num_steps=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        if self._rtc_enabled():
            raise NotImplementedError("PI05 IC inference does not support RTC yet")

        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = tokens.shape[0]
        device = tokens.device
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim),
                device,
            )

        if self.config.ic_sequence_mode == "xvla":
            prefix_embs, prefix_pad_masks, prefix_frame_ids = self.embed_prefix_ic_xvla(
                images_by_frame,
                img_masks_by_frame,
                tokens,
                masks,
                context_actions,
                next_images_by_frame,
                next_img_masks_by_frame,
            )
            prefix_key_frames = prefix_frame_ids[:, None, :]
            prefix_query_frames = prefix_frame_ids[:, :, None]
            prefix_att_2d_masks = (prefix_key_frames <= prefix_query_frames) & (
                prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
            )
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_ic(
                images_by_frame,
                img_masks_by_frame,
                tokens,
                masks,
                context_actions,
                next_images_by_frame,
                next_img_masks_by_frame,
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "sdpa"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "sdpa"  # noqa: SLF001

        past_key_values = copy.deepcopy(past_key_values)
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class PI05Policy(PreTrainedPolicy):
    """PI05 Policy for LeRobot."""

    config_class = PI05Config
    name = "pi05"

    def __init__(
        self,
        config: PI05Config,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Initialize the core PI05 model
        self.init_rtc_processor()
        self.model = PI05Pytorch(config, rtc_processor=self.rtc_processor)

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Override the from_pretrained method to handle key remapping and display important disclaimer."""
        print(
            "The PI05 model is a direct port of the OpenPI implementation. \n"
            "This implementation follows the original OpenPI structure for compatibility. \n"
            "Original implementation: https://github.com/Physical-Intelligence/openpi"
        )
        if pretrained_name_or_path is None:
            raise ValueError("pretrained_name_or_path is required")

        # Use provided config if available, otherwise create default config
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )

        # Initialize model without loading weights
        # Check if dataset_stats were provided in kwargs
        model = cls(config, **kwargs)

        # Load state dict (expects keys with "model." prefix)
        try:
            print(f"Loading model from: {pretrained_name_or_path}")
            try:
                from transformers.utils import cached_file

                resolved_file = cached_file(
                    pretrained_name_or_path,
                    "model.safetensors",
                    cache_dir=kwargs.get("cache_dir"),
                    force_download=kwargs.get("force_download", False),
                    resume_download=kwargs.get("resume_download"),
                    proxies=kwargs.get("proxies"),
                    token=kwargs.get("token"),
                    revision=kwargs.get("revision"),
                    local_files_only=kwargs.get("local_files_only", False),
                )
                from safetensors.torch import load_file

                original_state_dict = load_file(resolved_file)
                print("✓ Loaded state dict from model.safetensors")
            except Exception as e:
                print(f"Could not load state dict from remote files: {e}")
                print("Returning model without loading pretrained weights")
                return model

            # First, fix any key differences (see openpi model.py, _fix_pytorch_state_dict_keys)
            fixed_state_dict = model._fix_pytorch_state_dict_keys(original_state_dict, model.config)

            # Then add "model." prefix for all keys that don't already have it
            remapped_state_dict = {}
            remap_count = 0

            for key, value in fixed_state_dict.items():
                if not key.startswith("model."):
                    new_key = f"model.{key}"
                    remapped_state_dict[new_key] = value
                    remap_count += 1
                else:
                    remapped_state_dict[key] = value

            if remap_count > 0:
                print(f"Remapped {remap_count} state dict keys")

            embed_key = "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            if embed_key in remapped_state_dict:
                target_weight = model.state_dict()[embed_key]
                loaded_weight = remapped_state_dict[embed_key]
                if loaded_weight.shape != target_weight.shape and loaded_weight.shape[1:] == target_weight.shape[1:]:
                    expanded_weight = target_weight.clone()
                    expanded_weight[: loaded_weight.shape[0]].copy_(loaded_weight)
                    remapped_state_dict[embed_key] = expanded_weight

            # IC adds a new trainable ActionEmbedder, so old PI0.5 checkpoints are expected
            # to miss only those keys when IC is enabled.
            load_strict = strict and model.config.num_ic_frames <= 1
            missing_keys, unexpected_keys = model.load_state_dict(remapped_state_dict, strict=load_strict)

            if missing_keys:
                print(f"Missing keys when loading state dict: {len(missing_keys)} keys")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"  - {key}")
                else:
                    for key in missing_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(missing_keys) - 5} more")

            if unexpected_keys:
                print(f"Unexpected keys when loading state dict: {len(unexpected_keys)} keys")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"  - {key}")
                else:
                    for key in unexpected_keys[:5]:
                        print(f"  - {key}")
                    print(f"  ... and {len(unexpected_keys) - 5} more")

            if not missing_keys and not unexpected_keys:
                print("All keys loaded successfully!")

        except Exception as e:
            print(f"Warning: Could not load state dict: {e}")

        return model

    def _fix_pytorch_state_dict_keys(
        self, state_dict, model_config
    ):  # see openpi `BaseModelConfig, _fix_pytorch_state_dict_keys`
        """Fix state dict keys to match current model architecture."""
        import re

        fixed_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            # Handle layer norm structure changes: .weight -> .dense.weight + .dense.bias
            # For gemma expert layers
            if re.match(
                r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\.(input_layernorm|post_attention_layernorm)\.weight",
                key,
            ):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping layer norm key (adaRMS mismatch): {key}")
                    continue

            if re.match(r"paligemma_with_expert\.gemma_expert\.model\.norm\.weight", key):
                # Check if the model actually has adaRMS enabled for the expert
                expert_uses_adarms = getattr(
                    self.model.paligemma_with_expert.gemma_expert.config, "use_adarms", False
                )
                if expert_uses_adarms:
                    logging.warning(f"Skipping norm key (adaRMS mismatch): {key}")
                    continue

            # Handle MLP naming changes for pi05
            # pi05 model expects time_mlp_*, but checkpoint might have action_time_mlp_*
            if key.startswith("action_time_mlp_in."):
                new_key = key.replace("action_time_mlp_in.", "time_mlp_in.")
            elif key.startswith("action_time_mlp_out."):
                new_key = key.replace("action_time_mlp_out.", "time_mlp_out.")
            # Also handle state_proj which shouldn't exist in pi05
            if key.startswith("state_proj."):
                logging.warning(f"Skipping state_proj key in pi05 mode: {key}")
                continue

            # Handle vision tower embedding layer potential differences
            if "patch_embedding" in key:
                # Some checkpoints might have this, but current model expects different structure
                logging.warning(f"Vision embedding key might need handling: {key}")

            if (
                key == "model.paligemma_with_expert.paligemma.lm_head.weight"
                or key == "paligemma_with_expert.paligemma.lm_head.weight"
            ):
                fixed_state_dict[
                    "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
                ] = value.clone()

            fixed_state_dict[new_key] = value

        return fixed_state_dict

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        """Reset internal state - called when environment resets."""
        ic_tokenizer = getattr(self, "_ic_tokenizer", None)
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._ic_history = deque(maxlen=max(0, self.config.num_ic_frames - 1))
        self._ic_tokenizer = ic_tokenizer

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Create processor if config provided
        # If RTC is not enabled - we can still track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _prepare_image_tensor(self, img: Tensor) -> Tensor:
        device = next(self.parameters()).device
        if img.device != device:
            img = img.to(device)

        if img.dtype != torch.float32:
            img = img.to(torch.float32)

        is_channels_first = img.shape[1] == 3
        if is_channels_first:
            img = img.permute(0, 2, 3, 1)

        if img.shape[1:3] != self.config.image_resolution:
            img = resize_with_pad_torch(img, *self.config.image_resolution)

        img = img * 2.0 - 1.0

        if is_channels_first:
            img = img.permute(0, 3, 1, 2)
        return img

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.

        Images from LeRobot are typically in [B, C, H, W] format and normalized to [0, 1].
        PaliGemma expects images in [B, C, H, W] format and normalized to [-1, 1].
        """
        images = []
        img_masks = []

        # Get device from model parameters
        device = next(self.parameters()).device

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = self._prepare_image_tensor(batch[key])

            images.append(img)
            # Create mask (all ones for real images)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        # Create image features not present in the batch as fully 0 padded images
        for _num_empty_cameras in range(len(missing_img_keys)):
            img = torch.ones_like(img) * -1  # Padded with -1 for SigLIP
            mask = torch.zeros_like(mask)  # Mask is zero for empty cameras
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def _preprocess_images_ic(self, batch: dict[str, Tensor]) -> tuple[list[list[Tensor]], list[list[Tensor]]]:
        images_by_key = []
        masks_by_key = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        device = next(self.parameters()).device
        num_frames = None
        for key in present_img_keys:
            img = batch[key]
            if img.ndim != 5:
                raise ValueError(f"IC image feature {key} must have shape [B,K,C,H,W], got {img.shape}")
            bsize, frames = img.shape[:2]
            if num_frames is None:
                num_frames = frames
            elif frames != num_frames:
                raise ValueError(f"IC image feature {key} has {frames} frames, expected {num_frames}")
            flat = img.reshape(bsize * frames, *img.shape[2:])
            flat = self._prepare_image_tensor(flat)
            images_by_key.append(flat.reshape(bsize, frames, *flat.shape[1:]))
            masks_by_key.append(torch.ones(bsize, frames, dtype=torch.bool, device=device))

        for _ in missing_img_keys:
            img = torch.ones_like(images_by_key[-1]) * -1
            mask = torch.zeros_like(masks_by_key[-1])
            images_by_key.append(img)
            masks_by_key.append(mask)

        return (
            [[images[:, frame_idx] for images in images_by_key] for frame_idx in range(num_frames)],
            [[masks[:, frame_idx] for masks in masks_by_key] for frame_idx in range(num_frames)],
        )

    def _preprocess_next_images_ic(
        self, batch: dict[str, Tensor], num_context_frames: int
    ) -> tuple[list[list[Tensor]], list[list[Tensor]]]:
        if not self.config.include_next_obs:
            return None, None

        next_images_by_key = []
        next_masks_by_key = []
        device = next(self.parameters()).device
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features: {self.config.image_features})"
            )

        for key in present_img_keys:
            next_key = next_obs_key(key)
            if next_key not in batch:
                raise ValueError(f"PI05 IC include_next_obs=True requires '{next_key}' in the batch")
            img = batch[next_key]
            if img.ndim != 5:
                raise ValueError(f"IC next image feature {next_key} must have shape [B,K-1,C,H,W], got {img.shape}")
            bsize, frames = img.shape[:2]
            if frames != num_context_frames:
                raise ValueError(
                    f"IC next image feature {next_key} has {frames} frames, expected {num_context_frames}"
                )
            flat = img.reshape(bsize * frames, *img.shape[2:])
            flat = self._prepare_image_tensor(flat)
            next_images_by_key.append(flat.reshape(bsize, frames, *flat.shape[1:]))
            next_masks_by_key.append(torch.ones(bsize, frames, dtype=torch.bool, device=device))

        for _ in missing_img_keys:
            img = torch.ones_like(next_images_by_key[-1]) * -1
            mask = torch.zeros_like(next_masks_by_key[-1])
            next_images_by_key.append(img)
            next_masks_by_key.append(mask)

        return (
            [[images[:, frame_idx] for images in next_images_by_key] for frame_idx in range(num_context_frames)],
            [[masks[:, frame_idx] for masks in next_masks_by_key] for frame_idx in range(num_context_frames)],
        )

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _ic_enabled(self) -> bool:
        return self.config.num_ic_frames > 1

    def _split_ic_actions(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        actions = self._reshape_ic_actions(actions)
        return actions[:, :-1], actions[:, -1]

    def _reshape_ic_actions(self, actions: Tensor) -> Tensor:
        bsize, total_steps, dim = actions.shape
        expected_steps = self.config.num_ic_frames * self.config.chunk_size
        if total_steps != expected_steps:
            raise ValueError(
                f"PI05 IC expected action length {expected_steps}, got {total_steps}. "
                "Check action_delta_indices and dataset sampling."
            )
        return actions.reshape(bsize, self.config.num_ic_frames, self.config.chunk_size, dim)

    def _extract_task_text(self, task) -> str:
        if isinstance(task, list):
            task = task[0]
        task = str(task)
        if task.startswith("Task:"):
            task = task.removeprefix("Task:").strip()
            for separator in [", State:", "; Demo", "; Query State:"]:
                if separator in task:
                    task = task.split(separator, 1)[0]
                    break
        return task

    def _get_ic_tokenizer(self):
        if getattr(self, "_ic_tokenizer", None) is None:
            from transformers import AutoTokenizer

            self._ic_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            if self.config.ic_sequence_mode == "xvla":
                self._ic_tokenizer.add_special_tokens(
                    {"additional_special_tokens": list(PI05_XVLA_SPECIAL_TOKENS)},
                    replace_additional_special_tokens=False,
                )
        return self._ic_tokenizer

    def _tokenize_ic_prompt(
        self, task: str, states: Tensor, next_states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        tokenizer = self._get_ic_tokenizer()
        discretized_states = _discretize_normalized_state(states)[0]
        next_discretized_states = (
            _discretize_normalized_state(next_states)[0] if next_states is not None else None
        )
        if self.config.ic_sequence_mode == "xvla":
            prompt = format_pi05_xvla_ic_prompt(
                task,
                discretized_states,
                num_action_tokens=self.config.num_action_tokens,
                include_next_obs=self.config.include_next_obs,
                use_frame_sep=self.config.use_frame_sep,
                next_discretized_states=next_discretized_states,
            )
        else:
            prompt = format_pi05_ic_prompt(
                task,
                discretized_states,
                include_next_obs=self.config.include_next_obs,
                next_discretized_states=next_discretized_states,
            )
        tokenized = tokenizer(
            [prompt],
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        return (
            tokenized["input_ids"].to(device),
            tokenized["attention_mask"].to(device, dtype=torch.bool),
        )

    def _tokenize_ic_frame_prompts(
        self, tasks, states: Tensor, next_states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        tokenizer = self._get_ic_tokenizer()

        bsize, num_frames = states.shape[:2]
        if isinstance(tasks, str):
            tasks = [tasks] * bsize
        elif len(tasks) == 1 and bsize > 1:
            tasks = list(tasks) * bsize
        tasks = [self._extract_task_text(tasks[i]) for i in range(bsize)]

        discretized_states = _discretize_normalized_state(states)
        next_discretized_states = (
            _discretize_normalized_state(next_states) if next_states is not None else None
        )
        prompts = [
            format_pi05_ic_prompt(
                tasks[batch_idx],
                discretized_states[batch_idx, : frame_idx + 1],
                include_next_obs=self.config.include_next_obs,
                next_discretized_states=(
                    next_discretized_states[batch_idx, :frame_idx]
                    if next_discretized_states is not None
                    else None
                ),
            )
            for batch_idx in range(bsize)
            for frame_idx in range(num_frames)
        ]
        tokenized = tokenizer(
            prompts,
            max_length=self.config.tokenizer_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        return (
            tokenized["input_ids"].reshape(bsize, num_frames, -1).to(device),
            tokenized["attention_mask"].reshape(bsize, num_frames, -1).to(device, dtype=torch.bool),
        )

    def _select_ic_history_entries(self) -> list[dict] | None:
        num_context = self.config.num_ic_frames - 1
        if len(self._ic_history) < num_context:
            return None
        history = list(self._ic_history)
        stride = self.config.ic_stride or 1
        if self.config.ic_demo_mode == "fixed_first":
            return [history[0]] * num_context
        if self.config.ic_demo_mode == "early":
            indices = list(range(0, len(history), stride))[:num_context]
            return [history[i] for i in indices] if len(indices) == num_context else None

        indices = [len(history) - 1 - i * stride for i in range(num_context)]
        if indices[-1] < 0:
            return None
        indices.reverse()
        return [history[i] for i in indices]

    def _build_ic_inference_inputs(self, batch: dict[str, Tensor]):
        if not self._ic_enabled():
            return None
        if OBS_STATE not in batch or batch[OBS_STATE].shape[0] != 1:
            return None

        entries = self._select_ic_history_entries()
        if entries is None:
            return None

        ic_batch = {}
        for key in self.config.image_features:
            if key not in batch:
                continue
            context_images = (
                [batch[key][0].detach()] * len(entries)
                if self.config.clone_query_as_demo
                else [entry["images"][key] for entry in entries]
            )
            ic_batch[key] = torch.stack(
                context_images + [batch[key][0].detach()],
                dim=0,
            ).unsqueeze(0)
            if self.config.include_next_obs:
                next_context_images = context_images[1:] + [batch[key][0].detach()]
                ic_batch[next_obs_key(key)] = torch.stack(next_context_images, dim=0).unsqueeze(0)

        context_states = (
            [batch[OBS_STATE][0].detach()] * len(entries)
            if self.config.clone_query_as_demo
            else [entry["state"] for entry in entries]
        )
        states = torch.stack(context_states + [batch[OBS_STATE][0].detach()], dim=0).unsqueeze(0)
        if self.config.include_next_obs:
            ic_batch[next_obs_key(OBS_STATE)] = torch.stack(
                context_states[1:] + [batch[OBS_STATE][0].detach()],
                dim=0,
            ).unsqueeze(0)
        task = batch.get("task", [""])
        tokens, masks = self._tokenize_ic_prompt(
            self._extract_task_text(task),
            states,
            ic_batch.get(next_obs_key(OBS_STATE)),
        )

        context_actions = torch.stack([entry["action"] for entry in entries], dim=0).unsqueeze(0)
        context_actions = pad_vector(context_actions, self.config.max_action_dim)
        if self.config.zero_ic_actions:
            context_actions = torch.zeros_like(context_actions)

        images_by_frame, img_masks_by_frame = self._preprocess_images_ic(ic_batch)
        next_images_by_frame, next_img_masks_by_frame = self._preprocess_next_images_ic(
            ic_batch, self.config.num_ic_frames - 1
        )
        return (
            images_by_frame,
            img_masks_by_frame,
            tokens,
            masks,
            context_actions.to(tokens.device),
            next_images_by_frame,
            next_img_masks_by_frame,
        )

    def _record_ic_history(self, batch: dict[str, Tensor], actions: Tensor) -> None:
        if not self._ic_enabled() or actions.shape[0] != 1 or OBS_STATE not in batch:
            return
        if batch[OBS_STATE].shape[0] != 1:
            return

        action = actions[0].detach().clone()
        if self.config.n_action_steps < action.shape[0]:
            action[self.config.n_action_steps :] = 0

        images = {
            key: batch[key][0].detach().clone()
            for key in self.config.image_features
            if key in batch
        }
        if not images:
            return
        self._ic_history.append(
            {
                "images": images,
                "state": batch[OBS_STATE][0].detach().clone(),
                "action": action,
            }
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            full_actions = self.predict_action_chunk(batch)
            actions = full_actions[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        ic_inputs = self._build_ic_inference_inputs(batch)
        if ic_inputs is not None:
            (
                images_by_frame,
                img_masks_by_frame,
                ic_tokens,
                ic_masks,
                context_actions,
                next_images_by_frame,
                next_img_masks_by_frame,
            ) = ic_inputs
            actions = self.model.sample_actions_ic(
                images_by_frame,
                img_masks_by_frame,
                ic_tokens,
                ic_masks,
                context_actions,
                next_images_by_frame,
                next_img_masks_by_frame,
                **kwargs,
            )
        else:
            images, img_masks = self._preprocess_images(batch)
            # Sample actions using the model (pass through RTC kwargs, no separate state needed for PI05)
            actions = self.model.sample_actions(images, img_masks, tokens, masks, **kwargs)

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]
        self._record_ic_history(batch, actions)

        return actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training.

        Args:
            batch: Training batch containing observations and actions.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        # Prepare inputs
        tokens, masks = batch[f"{OBS_LANGUAGE_TOKENS}"], batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.prepare_action(batch)

        if self._ic_enabled():
            images_by_frame, img_masks_by_frame = self._preprocess_images_ic(batch)
            next_images_by_frame, next_img_masks_by_frame = self._preprocess_next_images_ic(
                batch, self.config.num_ic_frames - 1
            )
            if self.config.ic_loss_mode == "query_only":
                context_actions, query_actions = self._split_ic_actions(actions)
                losses = self.model.forward_ic_query_only(
                    images_by_frame,
                    img_masks_by_frame,
                    tokens,
                    masks,
                    context_actions,
                    query_actions,
                    next_images_by_frame,
                    next_img_masks_by_frame,
                )
            elif self.config.ic_loss_mode == "all_frames":
                if OBS_STATE not in batch:
                    raise ValueError("PI05 all-frame IC requires observation.state in the batch")
                ic_actions = self._reshape_ic_actions(actions)
                if self.config.ic_sequence_mode == "xvla":
                    tokens_by_frame, masks_by_frame = tokens, masks
                else:
                    tokens_by_frame, masks_by_frame = self._tokenize_ic_frame_prompts(
                        batch.get("task", [""] * actions.shape[0]),
                        batch[OBS_STATE],
                        batch.get(next_obs_key(OBS_STATE)),
                    )
                losses = self.model.forward_ic_all_frames(
                    images_by_frame,
                    img_masks_by_frame,
                    tokens_by_frame,
                    masks_by_frame,
                    ic_actions,
                    next_images_by_frame,
                    next_img_masks_by_frame,
                )
            else:
                raise ValueError(f"Invalid ic_loss_mode: {self.config.ic_loss_mode}")
        else:
            images, img_masks = self._preprocess_images(batch)
            # Compute loss (no separate state needed for PI05)
            losses = self.model.forward(images, img_masks, tokens, masks, actions)

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[..., :original_action_dim]
        raw_losses = losses
        if losses.ndim == 4 and self.config.ic_frame_loss_weights is not None:
            frame_weights = torch.tensor(
                self.config.ic_frame_loss_weights,
                dtype=losses.dtype,
                device=losses.device,
            )
            frame_weights = frame_weights / frame_weights.mean()
            losses = losses * frame_weights[None, :, None, None]

        loss_dict = {
            "loss_per_dim": losses.mean(dim=tuple(range(losses.ndim - 1))).detach().cpu().numpy().tolist(),
        }
        if raw_losses.ndim == 4:
            loss_dict["loss_per_frame"] = raw_losses.mean(dim=(0, 2, 3)).detach().cpu().numpy().tolist()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over time and action dims
            per_sample_loss = losses.mean(dim=tuple(range(1, losses.ndim)))
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss
            loss = losses.mean()
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for PI0.5 fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({common_projections}))"
        modules_to_save = ["model.action_embedder"] if self.config.num_ic_frames > 1 else []
        if (
            self.config.num_ic_frames > 1
            and self.config.ic_sequence_mode == "xvla"
            and self.config.use_frame_position_embed
        ):
            modules_to_save.append("model.frame_position_embed")
        return {
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }
