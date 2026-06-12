#!/usr/bin/env bash
set -euo pipefail

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot

OUTPUT_DIR="${OUTPUT_DIR:-ckpt/rlt_task03_ddp8gpu_finetune}"
LOG_FILE="${LOG_FILE:-$OUTPUT_DIR/cloud_job.log}"
mkdir -p "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
set -x

if [[ -d /VLA-Data/scripts/lianqing/.local ]]; then
  ln -sfn /VLA-Data/scripts/lianqing/.local /root/.local
fi

VENV="${VENV:-/VLA-Data/scripts/lianqing/projects/vla/X-VLA/.venv}"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/VLA-Data/scripts/lianqing/.cache/torch}"

if [[ -z "${LOAD_FROM:-}" ]]; then
  LOAD_FROM="$(find ckpt/rlt_task03_finetune -maxdepth 1 -type f -name 'checkpoint_[0-9]*.pt' | sort | tail -n 1)"
fi

python - <<'PY'
import torch
import torchvision
import pyarrow
from PIL import Image

print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
PY

NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
STEPS="${STEPS:-5000}" \
SAVE_EVERY="${SAVE_EVERY:-1000}" \
LOG_EVERY="${LOG_EVERY:-50}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
ACTOR_LR="${ACTOR_LR:-5e-5}" \
CRITIC_LR="${CRITIC_LR:-1e-4}" \
REF_DROPOUT_P="${REF_DROPOUT_P:-0.1}" \
BC_LAMBDA_INIT="${BC_LAMBDA_INIT:-0.6}" \
BC_LAMBDA_FINAL="${BC_LAMBDA_FINAL:-0.5}" \
BC_ANNEAL_STEPS="${BC_ANNEAL_STEPS:-5000}" \
OUTPUT_DIR="$OUTPUT_DIR" \
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-ddp8gpu}" \
LOAD_FROM="$LOAD_FROM" \
bash scripts/train_rlt_pi05_task03_8gpu.sh

ls -lh "$OUTPUT_DIR"
