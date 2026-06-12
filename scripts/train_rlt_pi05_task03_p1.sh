#!/usr/bin/env bash
set -euo pipefail

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot

export PYTHONPATH="${PYTHONPATH:-}:src"

DATASET_ROOT="${DATASET_ROOT:-/VLA-Data/scripts/lianqing/data/piper_dataset}"
DATASET_LIST="${DATASET_LIST:-/VLA-Data/scripts/lianqing/projects/vla/lerobot/trainset_config/rlt_inference.txt}"
INCLUDES="${INCLUDES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-ckpt/rlt_task03_paper_ball_p1}"
LOAD_FROM="${LOAD_FROM:-}"

STEPS="${STEPS:-3000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"

CHUNK_C="${CHUNK_C:-16}"
ACTION_STD="${ACTION_STD:-0.05}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
RESNET_WEIGHTS="${RESNET_WEIGHTS:-imagenet}"

FEATURE_DIM="${FEATURE_DIM:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-3}"
PROPRIO_HIDDEN="${PROPRIO_HIDDEN:-256}"

ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"

REF_DROPOUT_P="${REF_DROPOUT_P:-0.5}"
BC_LAMBDA_INIT="${BC_LAMBDA_INIT:-1.0}"
BC_LAMBDA_FINAL="${BC_LAMBDA_FINAL:-0.01}"
BC_ANNEAL_STEPS="${BC_ANNEAL_STEPS:-5000}"
BC_TARGET="${BC_TARGET:-action}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"

LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [[ -n "$DATASET_LIST" ]]; then
  EXTRA_ARGS+=(--dataset-list "$DATASET_LIST")
fi
if [[ -n "$LOAD_FROM" ]]; then
  EXTRA_ARGS+=(--load-from "$LOAD_FROM")
fi
if [[ -n "$INCLUDES" ]]; then
  for include in $INCLUDES; do
    EXTRA_ARGS+=(--include "$include")
  done
fi

python -m lerobot.rl.rlt_pi05.train \
  --dataset-root "$DATASET_ROOT" \
  "${EXTRA_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --chunk-C "$CHUNK_C" \
  --action-std "$ACTION_STD" \
  --image-size "$IMAGE_SIZE" \
  --resnet-weights "$RESNET_WEIGHTS" \
  --feature-dim "$FEATURE_DIM" \
  --hidden-dim "$HIDDEN_DIM" \
  --hidden-layers "$HIDDEN_LAYERS" \
  --proprio-hidden "$PROPRIO_HIDDEN" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --gamma "$GAMMA" \
  --tau "$TAU" \
  --ref-dropout-p "$REF_DROPOUT_P" \
  --bc-lambda-init "$BC_LAMBDA_INIT" \
  --bc-lambda-final "$BC_LAMBDA_FINAL" \
  --bc-anneal-steps "$BC_ANNEAL_STEPS" \
  --bc-target "$BC_TARGET" \
  --warmup-steps "$WARMUP_STEPS" \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY"
