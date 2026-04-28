#!/usr/bin/env bash
# =============================================================================
# train_value_head.sh — 训练 RECAP 分布式 Value Head
#
# 当前默认预设：
#   Qwen/Qwen3-VL-2B-Instruct + qwen_vl backbone
#   保守显存配置，适合作为第一轮 smoke/train baseline
#
# 自动适配单机单卡、单机多卡、火山云 MLP 多节点三种场景，均通过
# accelerate launch 启动。
#
# 【本地用法】
#   ./train_value_head.sh <dataset_list.txt> [选项...]
#
# 【火山云用法】控制台「自定义任务」选 PyTorchDDP，入口命令：
#   bash /path/to/train_value_head.sh <dataset_list.txt> [选项...]
#
# 选项:
#   --batch_size=N              每张 GPU 的批量大小，默认 1
#   --steps=N                   总训练步数，默认 10000
#   --lr=F                      学习率，默认 1e-4
#   --save_freq=N               保存检查点频率（步），默认 1000
#   --eval_freq=N               验证集评估频率（步），默认 500
#   --log_freq=N                日志打印频率（步），默认 20
#   --backbone_family=NAME      骨干家族，qwen_vl | paligemma，默认 qwen_vl
#   --pretrained_model_name=ID  预训练 VLM，默认 Qwen/Qwen3-VL-2B-Instruct
#   --vlm_variant=NAME          仅 paligemma 分支使用，gemma_300m | gemma_2b，默认 gemma_300m
#   --precision=NAME            训练精度，float32 | bfloat16，默认 bfloat16
#   --init_from_policy=PATH     用预训练 PI05 策略初始化 VLM 权重
#   --allow_random_init         允许从随机初始化 backbone 开始训练（不推荐）
#   --output_dir=DIR            输出目录，默认 ckpt/value_head_<dataset>_<timestamp>
#   --wandb_project=NAME        WandB 项目名（留空则不上传）
#   --wandb_run_name=NAME       WandB run 名（默认自动生成）
#   --gradient_checkpointing    启用梯度检查点（默认开启）
#   --resume=PATH               从指定 checkpoint 文件恢复训练
#   --num_workers=N             DataLoader workers 数，默认 2
#   --video_backend=NAME        视频解码后端，默认 pyav
# =============================================================================
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../../.." && pwd)"

# =============================================================================
# 默认训练参数
# =============================================================================
export http_proxy=http://100.68.175.233:3128
export https_proxy=http://100.68.175.233:3128

batch_size=1
steps=10000
lr="1e-4"
save_freq=5000
eval_freq=2500
log_freq=20
eval_episodes=5
backbone_family="qwen_vl"
pretrained_model_name="Qwen/Qwen3-VL-2B-Instruct"
vlm_variant="gemma_300m"
precision="bfloat16"
init_from_policy=""
allow_random_init=false
output_dir=""
wandb_project=""
wandb_run_name=""
gradient_checkpointing=true
resume_path=""
num_workers=2
video_backend="pyav"
dataset_list_file=""
extra_args=()

# =============================================================================
# 解析命令行参数
# =============================================================================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch_size=*)             batch_size="${1#*=}" ;;
    --batch_size)               batch_size="$2"; shift ;;
    --steps=*)                  steps="${1#*=}" ;;
    --steps)                    steps="$2"; shift ;;
    --lr=*)                     lr="${1#*=}" ;;
    --lr)                       lr="$2"; shift ;;
    --save_freq=*)              save_freq="${1#*=}" ;;
    --save_freq)                save_freq="$2"; shift ;;
    --eval_freq=*)              eval_freq="${1#*=}" ;;
    --eval_freq)                eval_freq="$2"; shift ;;
    --log_freq=*)               log_freq="${1#*=}" ;;
    --log_freq)                 log_freq="$2"; shift ;;
    --eval_episodes=*)          eval_episodes="${1#*=}" ;;
    --eval_episodes)            eval_episodes="$2"; shift ;;
    --backbone_family=*)        backbone_family="${1#*=}" ;;
    --backbone_family)          backbone_family="$2"; shift ;;
    --pretrained_model_name=*)  pretrained_model_name="${1#*=}" ;;
    --pretrained_model_name)    pretrained_model_name="$2"; shift ;;
    --vlm_variant=*)            vlm_variant="${1#*=}" ;;
    --vlm_variant)              vlm_variant="$2"; shift ;;
    --precision=*)              precision="${1#*=}" ;;
    --precision)                precision="$2"; shift ;;
    --init_from_policy=*)       init_from_policy="${1#*=}" ;;
    --init_from_policy)         init_from_policy="$2"; shift ;;
    --allow_random_init)        allow_random_init=true ;;
    --output_dir=*)             output_dir="${1#*=}" ;;
    --output_dir)               output_dir="$2"; shift ;;
    --wandb_project=*)          wandb_project="${1#*=}" ;;
    --wandb_project)            wandb_project="$2"; shift ;;
    --wandb_run_name=*)         wandb_run_name="${1#*=}" ;;
    --wandb_run_name)           wandb_run_name="$2"; shift ;;
    --gradient_checkpointing)   gradient_checkpointing=true ;;
    --resume=*)                 resume_path="${1#*=}" ;;
    --resume)                   resume_path="$2"; shift ;;
    --num_workers=*)            num_workers="${1#*=}" ;;
    --num_workers)              num_workers="$2"; shift ;;
    --video_backend=*)          video_backend="${1#*=}" ;;
    --video_backend)            video_backend="$2"; shift ;;
    -*)                         extra_args+=("$1") ;;
    *)
      [[ -z "${dataset_list_file}" ]] && dataset_list_file="$1" || extra_args+=("$1") ;;
  esac
  shift
done

# =============================================================================
# 检查必填参数
# =============================================================================
if [[ -z "${dataset_list_file}" ]]; then
  echo "Error: 必须提供 <dataset_list.txt>。" >&2
  echo "用法: $0 <dataset_list.txt> [选项...]" >&2
  exit 1
fi

if [[ ! -f "${dataset_list_file}" ]]; then
  echo "Error: 数据集列表文件 '${dataset_list_file}' 不存在。" >&2
  exit 1
fi

dataset_list_file="$(realpath "${dataset_list_file}")"
dataset_label="$(basename "${dataset_list_file%.*}")"

# =============================================================================
# 读取火山云 MLP 环境变量（多节点时由平台注入，本地时取默认值）
# =============================================================================
num_machines="${MLP_WORKER_NUM:-1}"
node_rank="${MLP_ROLE_INDEX:-0}"
master_addr="${MLP_WORKER_0_HOST:-127.0.0.1}"
raw_port="${MLP_WORKER_0_PORT:-29500}"
master_port="${raw_port%%,*}"

# GPU 数：通过 nvidia-smi 自动检测本机可用卡数
num_gpus_local=$(nvidia-smi -L 2>/dev/null | wc -l)
if [[ "${num_gpus_local}" -eq 0 ]]; then
  echo "Error: 未检测到 GPU，请确认 nvidia-smi 可用。" >&2
  exit 1
fi

num_processes=$(( num_gpus_local * num_machines ))

# =============================================================================
# 分布式网络配置（多节点时）
# =============================================================================
export MASTER_ADDR="${master_addr}"
export MASTER_PORT="${master_port}"

if [[ "${num_machines}" -gt 1 ]]; then
  nccl_if="${MLP_IFNAME:-${NCCL_SOCKET_IFNAME:-eth0}}"
  export NCCL_SOCKET_IFNAME="${nccl_if}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${nccl_if}}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
fi

# =============================================================================
# 派生变量
# =============================================================================
timestamp="$(date +%Y%m%d_%H%M%S)"
export http_proxy=http://100.68.175.233:3128; export https_proxy=http://100.68.175.233:3128
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "${HF_TOKEN}"
fi
if [[ -f "${repo_root}/.venv/bin/activate" ]]; then
  source "${repo_root}/.venv/bin/activate"
fi
export PYTHONPATH="${repo_root}/transformers_53/src:${repo_root}/src:${PYTHONPATH:-}"
if [[ -n "${output_dir}" ]]; then
  output_dir="${output_dir%/}_${timestamp}"
elif [[ -n "${resume_path}" ]]; then
  # 恢复训练时默认沿用 checkpoint 所在目录的上级，但仍追加新的时间戳后缀
  output_dir="$(dirname "$(dirname "${resume_path}")")_${timestamp}"
else
  output_dir="ckpt/value_head_${dataset_label}_${timestamp}"
fi

# WandB run name 默认格式：value_head_<dataset>_<时间戳>
if [[ -z "${wandb_run_name}" && -n "${wandb_project}" ]]; then
  wandb_run_name="value_head_${dataset_label}_${timestamp}"
fi

# accelerate mixed_precision 标志
if [[ "${precision}" == "bfloat16" ]]; then
  mixed_precision="bf16"
else
  mixed_precision="no"
fi

# =============================================================================
# 打印配置摘要（仅主节点）
# =============================================================================
if [[ "${node_rank}" -eq 0 ]]; then
  echo
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║             Value Head Training Configuration                ║"
  echo "╚══════════════════════════════════════════════════════════════╝"
  printf "  %-28s %s\n" "dataset_list_file"      "${dataset_list_file}"
  # 列出文件里的每条数据集路径
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    printf "  %-28s %s\n" "" "  → ${line}"
  done < "${dataset_list_file}"
  printf "  %-28s %s\n" "output_dir"             "${output_dir}"
  echo "  ──────────────────────────────────────────────────────────"
  printf "  %-28s %s\n" "num_machines"            "${num_machines}"
  printf "  %-28s %s\n" "num_gpus_local"          "${num_gpus_local}"
  printf "  %-28s %s\n" "total_processes"         "${num_processes}"
  if [[ "${num_machines}" -gt 1 ]]; then
    printf "  %-28s %s\n" "master"                "${master_addr}:${master_port}"
    printf "  %-28s %s\n" "NCCL_SOCKET_IFNAME"    "${NCCL_SOCKET_IFNAME:-<unset>}"
  fi
  echo "  ──────────────────────────────────────────────────────────"
  printf "  %-28s %s\n" "backbone_family"         "${backbone_family}"
  printf "  %-28s %s\n" "pretrained_model_name"   "${pretrained_model_name}"
  printf "  %-28s %s\n" "vlm_variant"             "${vlm_variant}"
  printf "  %-28s %s\n" "precision"               "${precision}"
  printf "  %-28s %s\n" "batch_size (per GPU)"    "${batch_size}"
  printf "  %-28s %s\n" "effective_batch_size"    "$(( batch_size * num_processes ))"
  printf "  %-28s %s\n" "steps"                   "${steps}"
  printf "  %-28s %s\n" "lr"                      "${lr}"
  printf "  %-28s %s\n" "save_freq"               "${save_freq}"
  printf "  %-28s %s\n" "eval_freq"               "${eval_freq}"
  printf "  %-28s %s\n" "gradient_checkpointing"  "${gradient_checkpointing}"
  [[ -n "${init_from_policy}" ]] && \
    printf "  %-28s %s\n" "init_from_policy"      "${init_from_policy}"
  printf "  %-28s %s\n" "allow_random_init"       "${allow_random_init}"
  [[ -n "${resume_path}" ]] && \
    printf "  %-28s %s\n" "resume"                "${resume_path}"
  if [[ -n "${wandb_project}" ]]; then
    printf "  %-28s %s\n" "wandb_project"         "${wandb_project}"
    printf "  %-28s %s\n" "wandb_run_name"        "${wandb_run_name}"
  else
    printf "  %-28s %s\n" "wandb"                 "off"
  fi
  [[ ${#extra_args[@]} -gt 0 ]] && \
    printf "  %-28s %s\n" "extra args"            "${extra_args[*]}"
  echo
fi

# =============================================================================
# 构建 train_value_function.py 参数
# =============================================================================
train_args=(
  "${script_dir}/train_value_function.py"
  --dataset_list_file="${dataset_list_file}"
  --backbone_family="${backbone_family}"
  --pretrained_model_name="${pretrained_model_name}"
  --vlm_variant="${vlm_variant}"
  --precision="${precision}"
  --batch_size="${batch_size}"
  --lr="${lr}"
  --steps="${steps}"
  --save_freq="${save_freq}"
  --eval_freq="${eval_freq}"
  --eval_episodes="${eval_episodes}"
  --log_freq="${log_freq}"
  --num_workers="${num_workers}"
  --video_backend="${video_backend}"
  --output_dir="${output_dir}"
)

[[ "${gradient_checkpointing}" == "true" ]] && \
  train_args+=(--gradient_checkpointing)

[[ -n "${init_from_policy}" ]] && \
  train_args+=(--init_from_policy="${init_from_policy}")

[[ "${allow_random_init}" == "true" ]] && \
  train_args+=(--allow_random_init)

[[ -n "${resume_path}" ]] && \
  train_args+=(--resume="${resume_path}")

if [[ -n "${wandb_project}" ]]; then
  train_args+=(
    --wandb_project="${wandb_project}"
    --wandb_run_name="${wandb_run_name}"
  )
fi

[[ ${#extra_args[@]} -gt 0 ]] && \
  train_args+=("${extra_args[@]}")

# =============================================================================
# 构建 accelerate launch 参数
# =============================================================================
accel_args=(
  --num_processes="${num_processes}"
  --num_machines="${num_machines}"
  --machine_rank="${node_rank}"
  --main_process_ip="${master_addr}"
  --main_process_port="${master_port}"
  --mixed_precision="${mixed_precision}"
)

[[ "${num_processes}" -gt 1 ]] && accel_args=(--multi_gpu "${accel_args[@]}")

# =============================================================================
# 启动训练
# =============================================================================
echo "[节点 ${node_rank}] 启动 Value Head 训练: ${num_machines} 节点 × ${num_gpus_local} GPU = ${num_processes} 总进程"
echo

exec accelerate launch "${accel_args[@]}" "${train_args[@]}"
