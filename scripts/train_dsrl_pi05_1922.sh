#!/usr/bin/env bash
set -euo pipefail

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot

export PYTHONPATH="${PYTHONPATH:-}:src"

DATASET_ROOT="${DATASET_ROOT:-/VLA-Data/scripts/lianqing/data/piper_dataset/dagger-20260526-1922-task03-put-the-paper-ball-into-the-yellow-trash-bin}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/dsrl_pi05_1922}"

STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
RESNET_WEIGHTS="${RESNET_WEIGHTS:-imagenet}"

HIDDEN_DIM="${HIDDEN_DIM:-256}"
FEATURE_DIM="${FEATURE_DIM:-256}"
ACTOR_LR="${ACTOR_LR:-1e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
ALPHA_LR="${ALPHA_LR:-3e-4}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
NOISE_BOUND="${NOISE_BOUND:-1.5}"

LOG_EVERY="${LOG_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-1000}"

mkdir -p "$OUTPUT_DIR"

python -m lerobot.rl.dsrl_pi05.train_sac \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --image-size "$IMAGE_SIZE" \
  --resnet-weights "$RESNET_WEIGHTS" \
  --hidden-dim "$HIDDEN_DIM" \
  --feature-dim "$FEATURE_DIM" \
  --actor-lr "$ACTOR_LR" \
  --critic-lr "$CRITIC_LR" \
  --alpha-lr "$ALPHA_LR" \
  --gamma "$GAMMA" \
  --tau "$TAU" \
  --noise-bound "$NOISE_BOUND" \
  --noise-reduction first \
  --log-every "$LOG_EVERY" \
  --save-every "$SAVE_EVERY"
