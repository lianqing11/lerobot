#!/usr/bin/env bash
# Train ACT policy on a local lerobot dataset.
# Automatically detects GPU count and launches multi-GPU training via accelerate.
#
# Usage:
#   ./train_act_auto.sh <dataset_root> [extra lerobot-train args ...]
#
# Examples:
#   # Single/multi-GPU (auto-detected), default batch_size=8
#   ./train_act_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged
#
#   # Override batch_size (per GPU)
#   ./train_act_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged --batch_size=16
#
#   # With wandb
#   ./train_act_auto.sh ~/data/piper_dataset/piper-debugPaperBall-merged --wandb.enable=true
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
job_name="${dataset_repo_id}"
output_dir="ckpt/${dataset_repo_id}"

# Detect available GPUs
num_gpus=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
if [[ "${num_gpus}" -eq 0 ]]; then
  echo "Warning: No CUDA GPUs detected, training on CPU" >&2
fi

echo "=== Training Config ==="
echo "  dataset.root    = ${dataset_root}"
echo "  dataset.repo_id = ${dataset_repo_id}"
echo "  job_name        = ${job_name}"
echo "  output_dir      = ${output_dir}"
echo "  num_gpus        = ${num_gpus}"
if [[ ${#extra_args[@]} -gt 0 ]]; then
  echo "  extra args      = ${extra_args[*]}"
fi
echo

train_args=(
  --dataset.repo_id="${dataset_repo_id}"
  --dataset.root="${dataset_root}"
  --policy.type=act
  --policy.push_to_hub=false
  --log_freq=10
  --output_dir="${output_dir}"
  --job_name="${job_name}"
  --save_freq=1000
  "${extra_args[@]}"
)

if [[ "${num_gpus}" -gt 1 ]]; then
  echo "Launching multi-GPU training with ${num_gpus} GPUs via accelerate..."
  accelerate launch --num_processes="${num_gpus}" \
    -m lerobot.scripts.lerobot_train "${train_args[@]}"
else
  echo "Launching single-GPU training..."
  lerobot-train "${train_args[@]}"
fi
