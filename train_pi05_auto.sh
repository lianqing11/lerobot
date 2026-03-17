#!/usr/bin/env bash
# =============================================================================
# train_pi05.sh — Finetune π₀.₅ (Pi0.5) on a local LeRobot dataset.
#
# 自动适配单机单卡、单机多卡、火山云 MLP 多节点三种场景，均通过
# accelerate launch 启动。火山云多节点时平台自动注入 MLP_* 环境变量，
# 脚本无需额外参数。
#
# 【火山云用法】控制台「自定义任务」选 PyTorchDDP，入口命令：
#   bash /path/to/train_pi05.sh <dataset_root> [选项...]
#
# 【本地用法】
#   ./train_pi05.sh <dataset_root> [选项...]
#
# 用法:
#   ./train_pi05.sh <dataset_root>      单个数据集目录
#   ./train_pi05.sh <dataset_list.txt>  多数据集列表文件（每行一个路径）
#
# 选项:
#   --batch_size=N            每张 GPU 的批量大小，默认 8
#   --steps=N                 总训练步数，默认 50000
#   --save_freq=N             保存检查点频率（步），默认 1000
#   --log_freq=N              日志打印频率（步），默认 10
#   --pretrained_path=PATH    预训练模型路径或 Hub ID，默认 lerobot/pi05_base
#   --output_dir=DIR          输出目录，默认 ckpt/pi05_<dataset>_<timestamp>
#   --wandb / --no_wandb      开启/关闭 WandB（多卡时自动开启）
#   --gradient_checkpointing  启用梯度检查点（显存不足时使用）
#   --resume=DIR              从指定目录恢复训练
# =============================================================================
set -euo pipefail

# =============================================================================
# 默认训练参数
# =============================================================================
export http_proxy=http://100.68.175.233:3128; export https_proxy=http://100.68.175.233:3128
batch_size=16
steps=50000
save_freq=1000
log_freq=10
pretrained_path="lerobot/pi05_base"
output_dir=""
wandb_mode="auto"          # auto | on | off
gradient_checkpointing=false
resume_dir=""
dataset_input=""           # 单个目录 或 .txt 列表文件
extra_args=()

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch_size=*)           batch_size="${1#*=}" ;;
    --batch_size)             batch_size="$2"; shift ;;
    --steps=*)                steps="${1#*=}" ;;
    --steps)                  steps="$2"; shift ;;
    --save_freq=*)            save_freq="${1#*=}" ;;
    --save_freq)              save_freq="$2"; shift ;;
    --log_freq=*)             log_freq="${1#*=}" ;;
    --log_freq)               log_freq="$2"; shift ;;
    --pretrained_path=*)      pretrained_path="${1#*=}" ;;
    --pretrained_path)        pretrained_path="$2"; shift ;;
    --output_dir=*)           output_dir="${1#*=}" ;;
    --output_dir)             output_dir="$2"; shift ;;
    --wandb)                  wandb_mode="on" ;;
    --no_wandb)               wandb_mode="off" ;;
    --gradient_checkpointing) gradient_checkpointing=true ;;
    --resume=*)               resume_dir="${1#*=}" ;;
    --resume)                 resume_dir="$2"; shift ;;
    -*)                       extra_args+=("$1") ;;
    *)
      [[ -z "${dataset_input}" ]] && dataset_input="$1" || extra_args+=("$1") ;;
  esac
  shift
done

# =============================================================================
# 解析 dataset 输入：支持单个目录 或 .txt 列表文件
# =============================================================================
if [[ -z "${dataset_input}" ]]; then
  echo "Error: 必须提供 <dataset_root> 或 <dataset_list.txt>。" >&2
  echo "用法: $0 <dataset_root|list.txt> [选项...]" >&2
  exit 1
fi

# 判断是 txt 文件还是目录
use_list_file=false
dataset_root=""
dataset_repo_id=""
dataset_list_file=""

if [[ -f "${dataset_input}" ]]; then
  # ── txt 列表文件模式 ──────────────────────────────────────────────
  dataset_list_file="$(realpath "${dataset_input}")"
  use_list_file=true
  # job_name / output_dir 取文件名（去掉扩展名）作标识
  dataset_repo_id="$(basename "${dataset_list_file%.*}")"
elif [[ -d "${dataset_input}" ]]; then
  # ── 单目录模式 ────────────────────────────────────────────────────
  dataset_root="${dataset_input%/}"
  dataset_repo_id="$(basename "${dataset_root}")"
else
  echo "Error: '${dataset_input}' 既不是目录也不是文件。" >&2
  exit 1
fi

# =============================================================================
# 读取火山云 MLP 环境变量（多节点时由平台自动注入，本地运行时取默认值）
# =============================================================================
num_machines="${MLP_WORKER_NUM:-1}"
node_rank="${MLP_ROLE_INDEX:-0}"
master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}"
master_port="${MLP_WORKER_0_PORT:-29500}"

# GPU 数：优先使用平台注入值，否则用 nvidia-smi 检测
if [[ -n "${MLP_WORKER_GPU:-}" ]]; then
  num_gpus_local="${MLP_WORKER_GPU}"
else
  num_gpus_local=$(nvidia-smi -L 2>/dev/null | wc -l)
fi

num_processes=$(( num_gpus_local * num_machines ))

# =============================================================================
# WandB 开关（多卡自动开启）
# =============================================================================
use_wandb=false
[[ "${wandb_mode}" == "on" ]] && use_wandb=true
[[ "${wandb_mode}" == "auto" && "${num_processes}" -gt 1 ]] && use_wandb=true

# =============================================================================
# 派生变量
# =============================================================================
job_name="pi05_${dataset_repo_id}"
if [[ -n "${resume_dir}" ]]; then
  output_dir="${resume_dir}"
elif [[ -z "${output_dir}" ]]; then
  output_dir="ckpt/pi05_${dataset_repo_id}_$(date +%Y%m%d_%H%M%S)"
fi

# =============================================================================
# 打印配置摘要（仅主节点）
# =============================================================================
if [[ "${node_rank}" -eq 0 ]]; then
  echo
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║              Pi0.5 Finetune Configuration                    ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  printf "  %-28s %s\n" "dataset.repo_id"        "${dataset_repo_id}"
  if [[ "${use_list_file}" == "true" ]]; then
    printf "  %-28s %s\n" "dataset_list_file"    "${dataset_list_file}"
    # 打印文件中的每条数据集路径
    while IFS= read -r line || [[ -n "${line}" ]]; do
      [[ -z "${line}" || "${line}" == \#* ]] && continue
      printf "  %-28s %s\n" "" "  → ${line}"
    done < "${dataset_list_file}"
  else
    printf "  %-28s %s\n" "dataset.root"         "${dataset_root}"
  fi
  printf "  %-28s %s\n" "pretrained_path"        "${pretrained_path}"
  printf "  %-28s %s\n" "output_dir"             "${output_dir}"
  echo "  ──────────────────────────────────────────────────────────"
  printf "  %-28s %s\n" "num_machines"           "${num_machines}"
  printf "  %-28s %s\n" "num_gpus_local"         "${num_gpus_local}"
  printf "  %-28s %s\n" "total_processes"        "${num_processes}"
  [[ "${num_machines}" -gt 1 ]] && \
    printf "  %-28s %s\n" "master"               "${master_addr}:${master_port}"
  echo "  ──────────────────────────────────────────────────────────"
  printf "  %-28s %s\n" "batch_size (per GPU)"   "${batch_size}"
  printf "  %-28s %s\n" "steps"                  "${steps}"
  printf "  %-28s %s\n" "save_freq"              "${save_freq}"
  printf "  %-28s %s\n" "dtype"                  "bfloat16"
  printf "  %-28s %s\n" "gradient_checkpointing" "${gradient_checkpointing}"
  printf "  %-28s %s\n" "wandb"                  "${use_wandb}"
  [[ -n "${resume_dir}" ]] && \
    printf "  %-28s %s\n" "resume"               "${resume_dir}"
  [[ ${#extra_args[@]} -gt 0 ]] && \
    printf "  %-28s %s\n" "extra args"           "${extra_args[*]}"
  echo
fi

# =============================================================================
# 构建 lerobot-train 参数
# =============================================================================
if [[ "${use_list_file}" == "true" ]]; then
  dataset_args=(--dataset.dataset_list_file="${dataset_list_file}")
else
  dataset_args=(--dataset.repo_id="${dataset_repo_id}" --dataset.root="${dataset_root}")
fi

train_args=(
  "${dataset_args[@]}"
  --policy.type=pi05
  --policy.pretrained_path="${pretrained_path}"
  --policy.push_to_hub=false
  --policy.dtype=bfloat16
  --policy.device=cuda
  --output_dir="${output_dir}"
  --job_name="${job_name}"
  --batch_size="${batch_size}"
  --steps="${steps}"
  --save_freq="${save_freq}"
  --log_freq="${log_freq}"
  --wandb.enable="${use_wandb}"
)

[[ "${gradient_checkpointing}" == "true" ]] && \
  train_args+=(--policy.use_gradient_checkpointing=true)

[[ -n "${resume_dir}" ]] && \
  train_args+=(--resume=true)

[[ ${#extra_args[@]} -gt 0 ]] && \
  train_args+=("${extra_args[@]}")

# =============================================================================
# 启动（统一使用 accelerate launch）
# =============================================================================
echo "[节点 ${node_rank}] 启动训练: ${num_machines} 节点 × ${num_gpus_local} GPU = ${num_processes} 总进程"

accelerate launch \
  --num_processes="${num_processes}" \
  --num_machines="${num_machines}" \
  --machine_rank="${node_rank}" \
  --main_process_ip="${master_addr}" \
  --main_process_port="${master_port}" \
  --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train "${train_args[@]}"
