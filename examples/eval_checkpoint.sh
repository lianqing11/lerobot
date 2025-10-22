#!/bin/bash
# Evaluation script for ManiSkill checkpoints
# Usage: bash examples/eval_checkpoint.sh [checkpoint_path] [options]

set -e

# Default values
CHECKPOINT_PATH="${1:-outputs/act_StackCube-v2_2025-10-21_09-49/checkpoints/014000/pretrained_model}"
N_EPISODES="${2:-50}"
BATCH_SIZE="${3:-10}"

echo "=========================================="
echo "Evaluating ManiSkill Checkpoint"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Episodes: $N_EPISODES"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Check if checkpoint exists
if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    echo ""
    echo "Expected structure:"
    echo "  $CHECKPOINT_PATH/"
    echo "    ├── config.json"
    echo "    └── model.safetensors"
    exit 1
fi

# Create output directory for evaluation results
CHECKPOINT_NAME=$(basename $(dirname $(dirname $CHECKPOINT_PATH)))
STEP=$(basename $(dirname $CHECKPOINT_PATH))
OUTPUT_DIR="outputs/eval/${CHECKPOINT_NAME}_${STEP}_$(date +%Y%m%d_%H%M%S)"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Run evaluation on both tasks
lerobot-eval \
    --policy.path="$CHECKPOINT_PATH" \
    --env.type=maniskill \
    --env.task=StackCube-v2 \
    --env.obs_mode=rgb \
    --env.control_mode=pd_joint_pos \
    --env.sim_backend=cpu \
    --env.eval_tasks='[StackCube-pertube]' \
    --eval.n_episodes=$N_EPISODES \
    --eval.batch_size=$BATCH_SIZE \
    --policy.device=cuda \
    --output_dir="$OUTPUT_DIR" \
    --seed=1000

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Check the following files:"
echo "  - $OUTPUT_DIR/eval_info.json (metrics)"
echo "  - $OUTPUT_DIR/videos/ (evaluation videos)"
echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/eval_info.json | jq"
echo ""


