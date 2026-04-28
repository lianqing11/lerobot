#!/usr/bin/env python

# KAI-0-style advantage function for offline RL.
#
# The advantage function predicts the progress difference between two
# observations: A(o_ref, o_cur, ℓ) ∈ [-1, 1].
#
# Architecture:
#   - Same VLM backbone as the value head (Qwen-VL or PaliGemma)
#   - Two-timestep input: reference and current observations merged
#   - 3-layer MLP advantage head with SiLU + Tanh (following KAI-0)
#   - MSE loss on progress difference targets

import logging
import math
import os
from contextlib import contextmanager
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForImageTextToText

from lerobot.policies.pi05.configuration_pi05 import DEFAULT_IMAGE_SIZE
from lerobot.policies.pi05.modeling_pi05 import (
    get_gemma_config,
    make_att_2d_masks,
    resize_with_pad_torch,
)

logger = logging.getLogger(__name__)

DEFAULT_QWEN_VL_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def _hf_cache_snapshot_dir(model_id: str) -> Path | None:
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    cache_root = Path(hf_home) / "hub" / f"models--{model_id.replace('/', '--')}"
    ref_main = cache_root / "refs" / "main"
    if not ref_main.exists():
        return None
    snapshot_dir = cache_root / "snapshots" / ref_main.read_text().strip()
    return snapshot_dir if snapshot_dir.exists() else None


def _resolve_attn_implementation(attn_implementation: str | None) -> str | None:
    if attn_implementation in (None, "", "default"):
        return None
    if attn_implementation == "auto":
        if find_spec("flash_attn") is not None:
            return "flash_attention_2"
        if torch.cuda.is_available():
            return "sdpa"
        return None
    return attn_implementation


@contextmanager
def _without_socks_proxy():
    proxy_keys = ("ALL_PROXY", "all_proxy")
    saved = {key: os.environ.pop(key, None) for key in proxy_keys}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


class PI05AdvantageFunction(nn.Module):
    """KAI-0-style advantage function for offline RL.

    Predicts A(o_ref, o_cur, ℓ) ∈ [-1, 1] — the progress difference between
    a reference observation and a current observation, conditioned on instruction ℓ.

    Architecture:
        - VLM backbone (Qwen-VL or PaliGemma) processes merged two-timestep images
        - 3-layer MLP advantage head with SiLU activation and Tanh output
        - MSE loss on progress difference targets
    """

    def __init__(
        self,
        backbone_family: str = "qwen_vl",
        pretrained_model_name: str = DEFAULT_QWEN_VL_MODEL,
        vlm_variant: str = "gemma_300m",
        image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
        precision: str = "float32",
        gradient_checkpointing: bool = False,
        load_pretrained_backbone: bool = False,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.backbone_family = backbone_family
        self.pretrained_model_name = pretrained_model_name
        self.vlm_variant = vlm_variant
        self.image_resolution = image_resolution
        self.gradient_checkpointing_enabled = gradient_checkpointing
        self.attn_implementation = _resolve_attn_implementation(attn_implementation)

        if backbone_family == "paligemma":
            self._init_paligemma_backbone(vlm_variant, image_resolution)
        elif backbone_family == "qwen_vl":
            self._init_qwen_backbone(pretrained_model_name, load_pretrained_backbone, self.attn_implementation)
        else:
            raise ValueError(f"Unknown backbone_family: {backbone_family}")

        # Advantage head: 3-layer MLP with SiLU + Tanh (following KAI-0)
        self.advantage_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 1),
            nn.Tanh(),
        )

        self._apply_runtime_options(precision, gradient_checkpointing)

    # ── Backbone initialization ──────────────────────────────────────────

    def _init_paligemma_backbone(self, vlm_variant: str, image_resolution: tuple[int, int]) -> None:
        vlm_config = get_gemma_config(vlm_variant)

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

        self.vlm = PaliGemmaForConditionalGenerationWithPiGemma(config=vlm_config_hf)
        self.paligemma = self.vlm
        self.hidden_size = vlm_config.width

    def _init_qwen_backbone(
        self,
        pretrained_model_name: str,
        load_pretrained_backbone: bool,
        attn_implementation: str | None,
    ) -> None:
        cached_dir = _hf_cache_snapshot_dir(pretrained_model_name)
        if load_pretrained_backbone:
            logger.info("Loading pretrained Qwen-VL backbone from %s", pretrained_model_name)
            model_source = str(cached_dir) if cached_dir is not None else pretrained_model_name
            load_kwargs = {"low_cpu_mem_usage": True}
            if attn_implementation is not None:
                load_kwargs["attn_implementation"] = attn_implementation
            if cached_dir is not None:
                load_kwargs["local_files_only"] = True
            with _without_socks_proxy():
                self.vlm = AutoModelForImageTextToText.from_pretrained(model_source, **load_kwargs)
        else:
            logger.info("Initializing Qwen-VL backbone from config: %s", pretrained_model_name)
            config_source = str(cached_dir) if cached_dir is not None else pretrained_model_name
            config_kwargs = {"local_files_only": True} if cached_dir is not None else {}
            with _without_socks_proxy():
                config = AutoConfig.from_pretrained(config_source, **config_kwargs)
            if attn_implementation is not None:
                config._attn_implementation = attn_implementation
            self.vlm = AutoModelForImageTextToText.from_config(config)

        if attn_implementation is not None:
            logger.info("Qwen-VL attention implementation: %s", attn_implementation)

        hidden_size = getattr(self.vlm.config, "hidden_size", None)
        if hidden_size is None and hasattr(self.vlm.config, "text_config"):
            hidden_size = self.vlm.config.text_config.hidden_size
        if hidden_size is None:
            raise ValueError(f"Unable to resolve hidden_size for {pretrained_model_name}")
        self.hidden_size = int(hidden_size)

    def _apply_runtime_options(self, precision: str, gradient_checkpointing: bool) -> None:
        if gradient_checkpointing:
            if self.backbone_family == "paligemma":
                self.paligemma.model.language_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            else:
                self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            logger.info("Gradient checkpointing enabled")

        if precision != "bfloat16":
            return

        if self.backbone_family == "paligemma":
            self.paligemma.to(dtype=torch.bfloat16)
            for name, param in self.paligemma.named_parameters():
                if any(s in name for s in ["vision_tower", "multi_modal_projector"]):
                    param.data = param.data.to(dtype=torch.float32)
        else:
            self.vlm.to(dtype=torch.bfloat16)
        self.advantage_head.to(dtype=torch.float32)

    # ── PaliGemma forward helpers ────────────────────────────────────────

    def embed_image(self, image: torch.Tensor):
        if self.backbone_family != "paligemma":
            raise RuntimeError("embed_image is only used by the PaliGemma backbone")
        out_dtype = image.dtype
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image_outputs = self.vlm.model.get_image_features(image)
        features = image_outputs.pooler_output * self.vlm.config.text_config.hidden_size**0.5
        if features.dtype != out_dtype:
            features = features.to(out_dtype)
        return features

    def embed_language_tokens(self, tokens: torch.Tensor):
        if self.backbone_family != "paligemma":
            raise RuntimeError("embed_language_tokens is only used by the PaliGemma backbone")
        return self.vlm.model.language_model.embed_tokens(tokens)

    def forward_vlm(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        tokens: Tensor,
        masks: Tensor,
    ) -> Tensor:
        """Run images + language through VLM and return pooled hidden states."""
        if self.backbone_family != "paligemma":
            raise RuntimeError("forward_vlm is only used by the PaliGemma backbone")
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

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

        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        att_2d_masks_4d = torch.where(att_2d_masks_4d, 0.0, -1e15)

        model_dtype = self.vlm.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            embs = embs.to(dtype=torch.bfloat16)
            att_2d_masks_4d = att_2d_masks_4d.to(dtype=torch.bfloat16)

        output = self.vlm.model.language_model(
            inputs_embeds=embs,
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = output.last_hidden_state

        hidden_states = hidden_states.to(dtype=torch.float32)
        pad_masks_float = pad_masks.float().unsqueeze(-1)
        pooled = (hidden_states * pad_masks_float).sum(dim=1) / pad_masks_float.sum(dim=1).clamp(min=1.0)

        return pooled  # [B, hidden_size]

    # ── Image preprocessing ──────────────────────────────────────────────

    def preprocess_images(
        self, batch: dict[str, Tensor], image_features: list[str]
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images from a batch (same logic as PI05Policy)."""
        if self.backbone_family != "paligemma":
            raise RuntimeError("preprocess_images is only used by the PaliGemma backbone")
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

    # ── Two-timestep image merging ───────────────────────────────────────

    def _tensor_to_uint8(self, image: Tensor) -> np.ndarray:
        """Convert a [C, H, W] or [H, W, C] tensor in [0, 1] to uint8 numpy."""
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.permute(1, 2, 0)
        image = image.detach().cpu().to(torch.float32).clamp(0.0, 1.0)
        array = (image.numpy() * 255.0).round().astype(np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return array

    def _merge_twostep_images(
        self, batch: dict[str, Tensor], image_keys: list[str], ref_image_keys: list[str], batch_idx: int
    ) -> Image.Image:
        """Merge reference and current images into one composite PIL image.

        Layout: [ref_cam1, ref_cam2, ..., cur_cam1, cur_cam2, ...]
        """
        arrays = []
        # Reference images first
        for key in ref_image_keys:
            if key in batch:
                arrays.append(self._tensor_to_uint8(batch[key][batch_idx]))
        # Current images second
        for key in image_keys:
            if key in batch:
                arrays.append(self._tensor_to_uint8(batch[key][batch_idx]))
        merged = np.concatenate(arrays, axis=1)
        return Image.fromarray(merged)

    # ── Input preparation ────────────────────────────────────────────────

    def prepare_inputs(
        self,
        batch: dict[str, Tensor],
        image_keys: list[str],
        ref_image_keys: list[str],
        processor,
        prompts: list[str],
        device="cuda",
    ):
        """Build model inputs from two sets of image features (current + reference)."""
        if self.backbone_family == "paligemma":
            # PaliGemma: pass all images from both timesteps as separate entries
            all_image_keys = ref_image_keys + image_keys
            images, img_masks = self.preprocess_images(batch, all_image_keys)
            encoded = processor(
                prompts, padding="max_length", max_length=200, truncation=True, return_tensors="pt",
            )
            return {
                "images": images,
                "img_masks": img_masks,
                "tokens": encoded["input_ids"].to(device),
                "masks": encoded["attention_mask"].to(device).bool(),
            }

        # Qwen-VL: merge all images into one composite image per sample
        return self._build_qwen_inputs_twostep(prompts, batch, image_keys, ref_image_keys, processor, device)

    def _build_qwen_inputs_twostep(
        self,
        prompts: list[str],
        batch: dict[str, Tensor],
        image_keys: list[str],
        ref_image_keys: list[str],
        processor,
        device,
    ):
        """Qwen-VL: merge two-timestep images into one wide composite image."""
        messages = []
        merged_images = []
        batch_size = len(prompts)

        for batch_idx in range(batch_size):
            merged_images.append(
                self._merge_twostep_images(batch, image_keys, ref_image_keys, batch_idx)
            )
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompts[batch_idx]},
                        ],
                    }
                ]
            )

        if hasattr(processor, "apply_chat_template"):
            texts = [
                processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
                for message in messages
            ]
        else:
            texts = prompts

        model_inputs = processor(text=texts, images=merged_images, padding=True, return_tensors="pt")
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
        }

    # ── Forward pass ─────────────────────────────────────────────────────

    def forward(
        self,
        prepared_inputs: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """Return advantage prediction [B, 1]."""
        if self.backbone_family == "paligemma":
            return self._forward_paligemma(prepared_inputs)
        return self._forward_qwen(prepared_inputs)

    def _forward_paligemma(self, prepared_inputs: dict[str, Tensor]) -> Tensor:
        pooled = self.forward_vlm(
            prepared_inputs["images"],
            prepared_inputs["img_masks"],
            prepared_inputs["tokens"],
            prepared_inputs["masks"],
        )
        return self.advantage_head(pooled)

    def _forward_qwen(self, prepared_inputs: dict[str, Tensor]) -> Tensor:
        outputs = self.vlm.model(
            **prepared_inputs,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state.to(torch.float32)
        pad_masks = prepared_inputs["attention_mask"].float().unsqueeze(-1)
        pooled = (hidden_states * pad_masks).sum(dim=1) / pad_masks.sum(dim=1).clamp(min=1.0)
        return self.advantage_head(pooled)

    # ── Prediction and loss ──────────────────────────────────────────────

    def predict_advantage(self, prepared_inputs: dict[str, Tensor]) -> Tensor:
        """Return advantage prediction [B] (squeezed scalar)."""
        return self.forward(prepared_inputs).squeeze(-1)

    def compute_loss(
        self,
        prepared_inputs: dict[str, Tensor],
        progress_target: Tensor,
    ) -> tuple[Tensor, dict]:
        """Compute MSE loss on progress difference target.

        Args:
            progress_target: [B] float tensor in [-1, 1]

        Returns:
            (loss, info_dict)
        """
        pred = self.forward(prepared_inputs).squeeze(-1)  # [B]
        loss = F.mse_loss(pred, progress_target)

        with torch.no_grad():
            mae = (pred - progress_target).abs().mean()

        info = {
            "loss": loss.item(),
            "mae": mae.item(),
            "pred_mean": pred.mean().item(),
            "target_mean": progress_target.mean().item(),
        }
        return loss, info
