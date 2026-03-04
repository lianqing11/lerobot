#!/usr/bin/env bash
# ============================================================================
# Pi0.5 Training on Bridge (abs EEF)
# Supports: single-GPU / multi-GPU / multi-node
#
# Usage:
#   # Single GPU (auto-detected)
#   bash run_pi05_bridge.sh
#
#   # Multi-GPU on single node (auto-detected)
#   bash run_pi05_bridge.sh
#
#   # Multi-node: run on EACH node with proper env vars
#   # Node 0 (master):
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NUM_NODES=2 NODE_RANK=0 \
#     bash run_pi05_bridge.sh
#   # Node 1:
#   MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NUM_NODES=2 NODE_RANK=1 \
#     bash run_pi05_bridge.sh
#
#   # Override batch_size (per GPU), steps, etc via extra args:
#   bash run_pi05_bridge.sh --batch_size=8 --steps=100000
# ============================================================================
set -euo pipefail

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot
source .venv/bin/activate

# ---------- Configurable via env vars ----------
DATASET_ROOT="${DATASET_ROOT:-/VLA-Data/scripts/lianqing/data/lerobot/bridge_abs_eef}"
DATASET_REPO_ID="${DATASET_REPO_ID:-bridge_abs_eef}"
PRETRAINED="${PRETRAINED:-lerobot/pi05_base}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpt/pi05_bridge_abs_eef}-$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE="${BATCH_SIZE:-16}"
STEPS="${STEPS:-50000}"

NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------- Detect GPUs ----------
if command -v nvidia-smi &>/dev/null; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
else
  NUM_GPUS=1
fi

TOTAL_GPUS=$((NUM_NODES * NUM_GPUS))

# ---------- Wandb for multi-GPU ----------
WANDB_FLAG=()
if [[ "${TOTAL_GPUS}" -gt 1 ]]; then
  WANDB_FLAG=(--wandb.enable=true)
fi

echo
echo "=== Pi0.5 Bridge Training ==="
echo "  Dataset       : ${DATASET_ROOT}"
echo "  Pretrained    : ${PRETRAINED}"
echo "  Output        : ${OUTPUT_DIR}"
echo "  Batch size    : ${BATCH_SIZE} (per GPU)"
echo "  Steps         : ${STEPS}"
echo "  GPUs/node     : ${NUM_GPUS}"
echo "  Nodes         : ${NUM_NODES}"
echo "  Total GPUs    : ${TOTAL_GPUS}"
if [[ "${NUM_NODES}" -gt 1 ]]; then
  echo "  Master        : ${MASTER_ADDR}:${MASTER_PORT}"
  echo "  Node rank     : ${NODE_RANK}"
fi
echo

TRAIN_ARGS=(
  --dataset.repo_id="${DATASET_REPO_ID}"
  --dataset.root="${DATASET_ROOT}"
  --policy.type=pi05
  --policy.pretrained_path="${PRETRAINED}"
  --policy.push_to_hub=false
  --policy.dtype=bfloat16
  --policy.device=cuda
  --policy.gradient_checkpointing=true
  --output_dir="${OUTPUT_DIR}"
  --job_name=pi05_bridge
  --log_freq=10
  --batch_size="${BATCH_SIZE}"
  --save_freq=1000
  --steps="${STEPS}"
  "${WANDB_FLAG[@]}"
  "$@"
)

if [[ "${TOTAL_GPUS}" -le 1 ]]; then
  echo ">>> Single-GPU training"
  lerobot-train "${TRAIN_ARGS[@]}"
elif [[ "${NUM_NODES}" -eq 1 ]]; then
  echo ">>> Multi-GPU training (${NUM_GPUS} GPUs, single node)"
  accelerate launch \
    --num_processes="${NUM_GPUS}" \
    -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}"
else
  echo ">>> Multi-node training (${NUM_NODES} nodes x ${NUM_GPUS} GPUs = ${TOTAL_GPUS} total)"
  accelerate launch \
    --num_machines="${NUM_NODES}" \
    --num_processes="${TOTAL_GPUS}" \
    --machine_rank="${NODE_RANK}" \
    --main_process_ip="${MASTER_ADDR}" \
    --main_process_port="${MASTER_PORT}" \
    -m lerobot.scripts.lerobot_train "${TRAIN_ARGS[@]}"
fi
