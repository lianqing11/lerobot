#!/usr/bin/env bash
set -euo pipefail

cd /VLA-Data/scripts/lianqing/projects/vla/lerobot
source .venv/bin/activate

CHECKPOINT="${1:-ckpt/pi05_bridge_abs_eef-20260303_144945/checkpoints/023000/pretrained_model}"

python scripts/evaluate_widowx_pi05.py \
    --checkpoint "${CHECKPOINT}" \
    --output_dir eval_outputs/pi05_bridge_widowx \
    --episodes 24 \
    --save_video \
    --execute_steps 10 \
    "${@:2}"
