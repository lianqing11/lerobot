#!/bin/bash
# Example script for training a policy on ManiSkill StackCube-v1 task with evaluation
#
# Usage:
#   bash examples/train_maniskill_stackcube.sh <dataset_path>
#
# This script demonstrates how to train an ACT policy on the StackCube-v1
# task from ManiSkill with periodic evaluation during training.

# Check if dataset path is provided
# if [ -z "$1" ]; then
#     echo "Usage: $0 <dataset_repo_id>"
#     echo "Example: $0 /path/to/maniskill_dataset"
#     echo ""
#     echo "To create a dataset, first convert ManiSkill trajectories to LeRobot format:"
#     echo "  cd /root/projects/ManiSkill"
#     echo "  python -m mani_skill.trajectory.convert_to_lerobot \\"
#     echo "      trajectory.h5 output_dir --task-name StackCube-v1"
#     exit 1
# fi

DATASET_PATH=/VLA-Data/scripts/lianqing/projects/ManiSkill/mani_skill/trajectory/output/stackcube_v2_100
rm -rf outputs/train/maniskill_stackcube/
# Configuration
TASK="StackCube-v2"
OBS_MODE="rgb"
CONTROL_MODE="pd_ee_delta_pose"
POLICY_TYPE="act"
STEPS=100000
EVAL_FREQ=2000
SAVE_FREQ=2000
BATCH_SIZE=8
N_EVAL_EPISODES=10
EVAL_BATCH_SIZE=10
SIM_BACKEND="cpu"  # Change to "gpu" for faster evaluation
OUTPUT_DIR="outputs/train/maniskill_stackcube"

echo "=========================================="
echo "Training on ManiSkill StackCube-v1"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Task: $TASK"
echo "Policy: $POLICY_TYPE"
echo "Observation mode: $OBS_MODE"
echo "Control mode: $CONTROL_MODE"
echo "Steps: $STEPS"
echo "Eval frequency: $EVAL_FREQ"
echo "Save frequency: $SAVE_FREQ"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Run training with evaluation
lerobot-train \
    --dataset.repo_id=lianqing11/stack_cube_lerobot \
    --dataset.root=$DATASET_PATH \
    --env.type=maniskill \
    --env.task=$TASK \
    --env.obs_mode=$OBS_MODE \
    --env.control_mode=$CONTROL_MODE \
    --env.sim_backend=$SIM_BACKEND \
    --policy.type=$POLICY_TYPE \
    --policy.push_to_hub=false \
    --steps=$STEPS \
    --eval_freq=$EVAL_FREQ \
    --save_freq=$SAVE_FREQ \
    --batch_size=$BATCH_SIZE \
    --eval.n_episodes=$N_EVAL_EPISODES \
    --eval.batch_size=$EVAL_BATCH_SIZE \
    --output_dir=$OUTPUT_DIR \
    --wandb.enable=true \
    --wandb.project=maniskill_lerobot

echo "=========================================="
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "To evaluate the best checkpoint:"
echo "  bash examples/eval_maniskill_stackcube.sh $OUTPUT_DIR/checkpoints/best/pretrained_model"

