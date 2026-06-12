#!/usr/bin/env bash
set -euo pipefail

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot

export LOAD_FROM="${LOAD_FROM:-/VLA-Data/scripts/lianqing/projects/vla/lerobot/ckpt/rlt_task03_finetune_v2/checkpoint_018100.pt}"
export OUTPUT_DIR="${OUTPUT_DIR:-ckpt/rlt_task03_finetune_v2}"
export STEPS="${STEPS:-30000}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export ACTOR_LR="${ACTOR_LR:-1e-4}"
export CRITIC_LR="${CRITIC_LR:-1e-4}"
export REF_DROPOUT_P="${REF_DROPOUT_P:-0.2}"
export BC_TARGET="${BC_TARGET:-ref}"
export BC_LAMBDA_INIT="${BC_LAMBDA_INIT:-1.0}"
export BC_LAMBDA_FINAL="${BC_LAMBDA_FINAL:-0.3}"
export BC_ANNEAL_STEPS="${BC_ANNEAL_STEPS:-3000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export LOG_EVERY="${LOG_EVERY:-50}"

exec bash scripts/train_rlt_pi05_task03_p1.sh
