#!/usr/bin/env bash
set -euo pipefail

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot

export OUTPUT_DIR="${OUTPUT_DIR:-ckpt/rlt_task03_scratch}"
export STEPS="${STEPS:-3000}"
export WARMUP_STEPS="${WARMUP_STEPS:-300}"
export ACTOR_LR="${ACTOR_LR:-3e-4}"
export CRITIC_LR="${CRITIC_LR:-3e-4}"
export BC_TARGET="${BC_TARGET:-action}"
export BC_LAMBDA_INIT="${BC_LAMBDA_INIT:-1.0}"
export BC_LAMBDA_FINAL="${BC_LAMBDA_FINAL:-0.01}"
export BC_ANNEAL_STEPS="${BC_ANNEAL_STEPS:-3000}"
export SAVE_EVERY="${SAVE_EVERY:-1000}"
export LOG_EVERY="${LOG_EVERY:-50}"

exec bash scripts/train_rlt_pi05_task03_p1.sh
