#!/bin/bash
# 多GPU训练脚本
# 用法: bash examples/train_multigpu.sh --config stackcube_act.yaml --num_gpus 4

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 自动检测GPU数量
AUTO_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $AUTO_GPUS -eq 0 ]; then
    AUTO_GPUS=1  # 如果检测失败，默认为1
fi

# 默认参数
NUM_GPUS=${AUTO_GPUS}      # 自动检测的GPU数量
NUM_PROCESSES=8            # 默认进程数
MIXED_PRECISION="bf16"     # 默认使用bf16混合精度

# 帮助信息
if [ $# -eq 0 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    echo "用法: $0 --config <配置文件> [选项]"
    echo ""
    echo "检测到GPU数量: $AUTO_GPUS"
    echo ""
    echo "选项:"
    echo "  --num_gpus N         使用的GPU数量 (默认: $NUM_GPUS, 自动检测)"
    echo "  --num_processes N    进程数 (默认: $NUM_PROCESSES)"
    echo "  --gpu_ids IDS        指定GPU ID，例如 0,1,2,3"
    echo "  --mixed_precision    混合精度: fp16, bf16, no (默认: $MIXED_PRECISION)"
    echo ""
    echo "示例:"
    echo "  # 使用所有检测到的GPU (默认)"
    echo "  $0 --config stackcube_act.yaml"
    echo ""
    echo "  # 指定GPU数量和进程数"
    echo "  $0 --config stackcube_act.yaml --num_gpus 4 --num_processes 8"
    echo ""
    echo "  # 指定使用哪些GPU"
    echo "  $0 --config stackcube_act.yaml --gpu_ids 0,1,2,3"
    echo ""
    echo "  # 使用fp16混合精度"
    echo "  $0 --config stackcube_act.yaml --mixed_precision fp16"
    echo ""
    echo "可用配置:"
    ls -1 "$SCRIPT_DIR/configs"/*.yaml 2>/dev/null | xargs -n1 basename | sed 's/^/  /'
    exit 0
fi

# 解析参数
CONFIG_FILE=""
GPU_IDS=""
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config)
            shift
            CONFIG_FILE="$1"
            shift
            ;;
        --num_gpus)
            shift
            NUM_GPUS="$1"
            shift
            ;;
        --num_processes)
            shift
            NUM_PROCESSES="$1"
            shift
            ;;
        --gpu_ids)
            shift
            GPU_IDS="$1"
            shift
            ;;
        --mixed_precision)
            shift
            MIXED_PRECISION="$1"
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$CONFIG_FILE" ]; then
    echo "错误: 必须指定 --config 参数"
    exit 1
fi

# 查找配置文件
if [ -f "$CONFIG_FILE" ]; then
    FINAL_CONFIG="$CONFIG_FILE"
elif [ -f "$SCRIPT_DIR/configs/$CONFIG_FILE" ]; then
    FINAL_CONFIG="$SCRIPT_DIR/configs/$CONFIG_FILE"
else
    echo "错误: 找不到配置文件: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "多GPU训练"
echo "=========================================="
echo "配置文件: $FINAL_CONFIG"
echo "检测到GPU: $AUTO_GPUS 个"
echo "使用GPU数: $NUM_GPUS"
echo "进程数: $NUM_PROCESSES"
if [ -n "$GPU_IDS" ]; then
    echo "指定GPU: $GPU_IDS"
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    # 如果指定了GPU_IDS，重新计算NUM_GPUS
    NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)
    echo "实际使用: $NUM_GPUS 个GPU"
fi
echo "混合精度: $MIXED_PRECISION"
echo "=========================================="
echo ""

# 运行 accelerate launch
# 如果只有1个GPU，不使用多GPU模式
if [ $AUTO_GPUS -eq 1 ]; then
    echo "单GPU模式"
    lerobot-train --config_path "$FINAL_CONFIG" "${EXTRA_ARGS[@]}"
else
    echo "多GPU模式"
    accelerate launch \
        --multi_gpu \
        --num_processes $NUM_PROCESSES \
        --num_machines 1 \
        --mixed_precision $MIXED_PRECISION \
        -m lerobot.scripts.lerobot_train \
        --config_path "$FINAL_CONFIG" \
        "${EXTRA_ARGS[@]}"
fi

