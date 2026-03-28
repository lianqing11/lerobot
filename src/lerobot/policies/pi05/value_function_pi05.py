#!/usr/bin/env python

# Value function for RECAP (π*0.6 style advantage-conditioned RL)
# Based on: "π*0.6: a VLA That Learns From Experience" (Physical Intelligence)
#
# The value function uses the same VLM architecture as the policy (PaliGemma)
# but with a smaller backbone and no action expert. Instead, a classification
# head predicts a distribution over B=201 discretized value bins.

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE, PI05Config
from lerobot.policies.pi05.modeling_pi05 import (
    PaliGemmaWithExpertModel,
    get_gemma_config,
    make_att_2d_masks,
    resize_with_pad_torch,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

logger = logging.getLogger(__name__)


class PI05ValueFunction(nn.Module):
    """Distributional value function for RECAP-style offline RL.

    Predicts p_ϕ(V | o_t, ℓ) as a categorical distribution over B bins.
    The value represents the (negative) normalized number of steps to success.
    Bins span [-1, 0], where 0 = immediate success, -1 = maximum episode length.

    Architecture:
        - PaliGemma VLM (SigLIP vision encoder + Gemma language model)
        - No action expert (unlike the policy)
        - MLP value head → B bins with cross-entropy training
    """

    NUM_BINS = 201

    def __init__(
        self,
        vlm_variant: str = "gemma_300m",
        image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        precision: str = "float32",
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.vlm_variant = vlm_variant
        self.image_resolution = image_resolution
        self.gradient_checkpointing_enabled = gradient_checkpointing

        vlm_config = get_gemma_config(vlm_variant)

        # We only need the VLM, not the action expert.
        # Reuse PaliGemmaWithExpertModel but with a tiny dummy expert
        # that we won't use. Alternatively, build the VLM directly.
        # For simplicity, we instantiate the full PaliGemma VLM only.
        from lerobot.policies.pi_gemma import PaliGemmaForConditionalGenerationWithPiGemma
        from transformers.models.auto import CONFIG_MAPPING

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
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.image_size = image_resolution[0]
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = vlm_config.width
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.dtype = "float32"

        self.paligemma = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.hidden_size = vlm_config.width

        # Value head: hidden → bins
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.NUM_BINS),
        )

        # Bin centers for extracting continuous value from distribution
        bin_values = torch.linspace(-1.0, 0.0, self.NUM_BINS)
        self.register_buffer("bin_values", bin_values)

        if precision == "bfloat16":
            self.paligemma.to(dtype=torch.bfloat16)
            # Keep vision path in float32
            for name, param in self.paligemma.named_parameters():
                if any(s in name for s in ["vision_tower", "multi_modal_projector"]):
                    param.data = param.data.to(dtype=torch.float32)
            # Value head stays float32
            self.value_head.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
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

    def forward_vlm(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Run images + language through VLM and return pooled hidden states."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        # Process language tokens
        lang_emb = self.embed_language_tokens(tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks_t = att_masks_t[None, :].expand(bsize, len(att_masks))

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks_t)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # 4D attention mask
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        att_2d_masks_4d = torch.where(att_2d_masks_4d, 0.0, -1e15)

        if (
            self.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            embs = embs.to(dtype=torch.bfloat16)

        output = self.paligemma.model.language_model(
            inputs_embeds=embs,
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = output.last_hidden_state  # [B, seq_len, hidden]

        # Mean pool over valid (non-padding) tokens
        hidden_states = hidden_states.to(dtype=torch.float32)
        pad_masks_float = pad_masks.float().unsqueeze(-1)  # [B, seq_len, 1]
        pooled = (hidden_states * pad_masks_float).sum(dim=1) / pad_masks_float.sum(dim=1).clamp(min=1.0)

        return pooled  # [B, hidden_size]

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Return value distribution logits [B, NUM_BINS]."""
        pooled = self.forward_vlm(images, img_masks, tokens, masks)
        logits = self.value_head(pooled)
        return logits

    def predict_value(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Return continuous value estimate [B]."""
        logits = self.forward(images, img_masks, tokens, masks)
        probs = F.softmax(logits, dim=-1)
        value = (probs * self.bin_values).sum(dim=-1)
        return value

    def compute_loss(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
        target_bin_indices: Tensor,
    ) -> tuple[Tensor, dict]:
        """Compute cross-entropy loss on discretized returns.

        Args:
            target_bin_indices: [B] long tensor, bin index in [0, NUM_BINS-1]

        Returns:
            (loss, info_dict)
        """
        logits = self.forward(images, img_masks, tokens, masks)
        loss = F.cross_entropy(logits, target_bin_indices)

        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            pred_value = (probs * self.bin_values).sum(dim=-1)
            target_value = self.bin_values[target_bin_indices]
            mae = (pred_value - target_value).abs().mean()

        info = {
            "loss": loss.item(),
            "mae": mae.item(),
            "pred_value_mean": pred_value.mean().item(),
            "target_value_mean": target_value.mean().item(),
        }
        return loss, info

    def preprocess_images(
        self, batch: dict[str, Tensor], image_features: list[str]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images from a batch (same logic as PI05Policy)."""
        images = []
        img_masks = []
        device = next(self.parameters()).device

        for key in image_features:
            if key not in batch:
                continue
            img = batch[key]
            if img.device != device:
                img = img.to(device)
            if img.dtype != torch.float32:
                img = img.to(torch.float32)

            is_channels_first = img.shape[1] == 3
            if is_channels_first:
                img = img.permute(0, 2, 3, 1)

            if img.shape[1:3] != self.image_resolution:
                img = resize_with_pad_torch(img, *self.image_resolution)

            img = img * 2.0 - 1.0
            if is_channels_first:
                img = img.permute(0, 3, 1, 2)

            images.append(img)
            bsize = img.shape[0]
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            img_masks.append(mask)

        return images, img_masks


def returns_to_bin_indices(returns: Tensor, num_bins: int = 201) -> Tensor:
    """Discretize normalized returns in [-1, 0] to bin indices in [0, num_bins-1]."""
    clamped = returns.clamp(-1.0, 0.0)
    indices = ((clamped + 1.0) * (num_bins - 1)).long().clamp(0, num_bins - 1)
    return indices


def compute_episode_returns(
    episode_length: int,
    success: bool = True,
    c_fail: float = 100.0,
    max_episode_length: int | None = None,
) -> Tensor:
    """Compute per-timestep returns for one episode following Eq. 5 of the paper.

    Returns:
        Tensor of shape [episode_length] with normalized returns in [-1, 0].
    """
    if max_episode_length is None:
        max_episode_length = episode_length

    T = episode_length
    # r_t = -1 for t < T, r_T = 0 (success) or -c_fail (failure)
    # R_t = sum_{t'=t}^{T} r_{t'}
    if success:
        # R_t = -(T - 1 - t) for t = 0..T-2, R_{T-1} = 0
        returns = torch.arange(T, dtype=torch.float32)
        returns = -(T - 1 - returns)
        returns[-1] = 0.0
    else:
        returns = torch.arange(T, dtype=torch.float32)
        returns = -(T - 1 - returns) - c_fail
        returns[-1] = -c_fail

    # Normalize by max_episode_length to [-1, 0] range
    returns = returns / max_episode_length
    returns = returns.clamp(-1.0, 0.0)

    return returns
