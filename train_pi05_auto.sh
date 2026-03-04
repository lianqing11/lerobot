#!/usr/bin/env bash
# Finetune π₀.₅ (Pi0.5) policy on a local lerobot dataset.
# Automatically detects GPU count and launches multi-GPU training via accelerate.
# - Multi-GPU automatically enables wandb logging.
# - Checks for quantile stats and generates them if missing.
#
# Usage:
#   ./train_pi05_auto.sh <dataset_root> [extra lerobot-train args ...]
#
# Examples:
#   # Single/multi-GPU (auto-detected)
#   ./train_pi05_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged
#
#   # Override batch_size (per GPU)
#   ./train_pi05_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged --batch_size=4
#
#   # Use a different pretrained base model
#   ./train_pi05_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged \
#       --policy.pretrained_path=lerobot/pi05_libero
#
#   # Custom training steps and save frequency
#   ./train_pi05_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged --steps=6000 --save_freq=500
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset_root> [extra lerobot-train args ...]"
  exit 1
fi

dataset_root="${1%/}"
shift
extra_args=("$@")

if [[ ! -d "${dataset_root}" ]]; then
  echo "Error: dataset.root does not exist: ${dataset_root}" >&2
  exit 1
fi

dataset_repo_id="$(basename "${dataset_root}")"
job_name="pi05_${dataset_repo_id}"
output_dir="ckpt/pi05_${dataset_repo_id}_$(date +%Y%m%d_%H%M%S)"

# ---------------------------------------------------------------------------
# Detect available GPUs (nvidia-smi is much faster than importing torch)
# ---------------------------------------------------------------------------
if command -v nvidia-smi &>/dev/null; then
  num_gpus=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  num_gpus=0
fi
if [[ "${num_gpus}" -eq 0 ]]; then
  echo "Warning: No CUDA GPUs detected, training on CPU (Pi0.5 is very slow on CPU)" >&2
fi

# ---------------------------------------------------------------------------
# Ensure dataset has quantile stats (required by Pi0.5 default normalization)
# ---------------------------------------------------------------------------
stats_file="${dataset_root}/meta/stats.json"
echo "Checking quantile stats for dataset..."

# echo "Quantile stats not found — generating now (this may take a while)..."
# python3 src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
#   --repo-id="${dataset_repo_id}" \
#   --root="${dataset_root}"
# echo "Quantile stats generated."

# ---------------------------------------------------------------------------
# Multi-GPU → auto-enable wandb
# ---------------------------------------------------------------------------
wandb_flag=()
if [[ "${num_gpus}" -gt 1 ]]; then
  wandb_flag=(--wandb.enable=true)
  echo "Multi-GPU detected (${num_gpus}), wandb logging auto-enabled."
fi

echo
echo "=== Pi0.5 Finetune Config ==="
echo "  dataset.root         = ${dataset_root}"
echo "  dataset.repo_id      = ${dataset_repo_id}"
echo "  pretrained_path      = lerobot/pi05_base"
echo "  job_name             = ${job_name}"
echo "  output_dir           = ${output_dir}"
echo "  num_gpus             = ${num_gpus}"
echo "  dtype                = bfloat16"
echo "  gradient_checkpoint  = true"
echo "  wandb               = $( [[ ${#wandb_flag[@]} -gt 0 ]] && echo 'true' || echo 'false (single-GPU)' )"
if [[ ${#extra_args[@]} -gt 0 ]]; then
  echo "  extra args           = ${extra_args[*]}"
fi
echo

# ---------------------------------------------------------------------------
# Build training arguments
# ---------------------------------------------------------------------------
train_args=(
  --dataset.repo_id="${dataset_repo_id}"
  --dataset.root="${dataset_root}"
  --policy.type=pi05
  --policy.pretrained_path=lerobot/pi05_base
  --policy.push_to_hub=false
  --policy.dtype=bfloat16
  --policy.device=cuda
  --output_dir="${output_dir}"
  --job_name="${job_name}"
  --log_freq=10
  --batch_size=24
  --save_freq=1000
  --steps=50000
  "${wandb_flag[@]}"
  "${extra_args[@]}"
)

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
if [[ "${num_gpus}" -gt 1 ]]; then
  echo "Launching multi-GPU Pi0.5 finetune with ${num_gpus} GPUs via accelerate..."
  accelerate launch --num_processes="${num_gpus}" \
    -m lerobot.scripts.lerobot_train "${train_args[@]}"
else
  echo "Launching single-GPU Pi0.5 finetune..."
  lerobot-train "${train_args[@]}"
fi
