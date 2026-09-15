#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
MODEL_PATH=${MODEL_PATH:-${ROOT}/Qwen3-VL-8B-Instruct}
SFT_ROOT=${ROOT}/qwen3-vl-src/qwen-vl-finetune
OUTPUT_DIR=${OUTPUT_DIR:-${SFT_ROOT}/output/groundcua-qwen3-vl-8b-sft-zero2}
EVAL_SNAPSHOT_DIR=${EVAL_SNAPSHOT_DIR:-${OUTPUT_DIR}/eval_snapshots}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
GPU_IDS=${GPU_IDS:-0,1,2,3}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# Final recipe selected from the GroundCUA throughput/optimization sweep.
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-32}
NUM_EPOCHS=${NUM_EPOCHS:-2}
MAX_PIXELS=${MAX_PIXELS:-2359296}
MIN_PIXELS=${MIN_PIXELS:-262144}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-16}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-4}
GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL:-30}

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${LOG_DIR:-${SFT_ROOT}/training_logs/groundcua_sft_zero2_${STAMP}}
TRAIN_LOG=${LOG_DIR}/train.log
GPU_LOG=${LOG_DIR}/gpu.csv
RUN_INFO=${LOG_DIR}/run_info.txt
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${EVAL_SNAPSHOT_DIR}"

cd "${SFT_ROOT}"
export GROUND_CUA_SFT_JSONL=${GROUND_CUA_SFT_JSONL:-${ROOT}/datasets/groundcua/sft/qwen3vl_train.jsonl}

GLOBAL_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))

monitor_pid=""
cleanup_monitor() {
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}

notify_email() {
  local rc=${1:-0}
  if [[ -z "${NOTIFY_EMAIL:-}" ]]; then
    return 0
  fi
  if ! command -v mail >/dev/null 2>&1; then
    echo "NOTIFY_EMAIL is set, but 'mail' is unavailable; skipping notification." >&2
    return 0
  fi

  local host status subject
  host=$(hostname)
  if [[ ${rc} -eq 0 ]]; then
    status="COMPLETED"
  else
    status="FAILED (exit=${rc})"
  fi
  subject="[GroundCUA SFT] Full training ${status} - ${host}"

  {
    echo "GroundCUA Qwen3-VL-8B full SFT finished."
    echo
    echo "Host: ${host}"
    echo "Status: ${status}"
    echo "Finished: $(date --iso-8601=seconds)"
    echo "Output: ${OUTPUT_DIR}"
    echo "Evaluation snapshots: ${EVAL_SNAPSHOT_DIR}"
    echo "Training log: ${TRAIN_LOG}"
    echo "GPU log: ${GPU_LOG}"
    echo
    if [[ -f "${RUN_INFO}" ]]; then
      cat "${RUN_INFO}"
    fi
    echo
    echo "Last training log lines:"
    tail -40 "${TRAIN_LOG}" 2>/dev/null || true
  } | mail -s "${subject}" "${NOTIFY_EMAIL}" || true
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM
  cleanup_monitor
  notify_email "${rc}"
  exit "${rc}"
}
trap on_exit EXIT INT TERM

cat > "${RUN_INFO}" <<EOF
Started: $(date --iso-8601=seconds)
Model: ${MODEL_PATH}
Dataset: ${GROUND_CUA_SFT_JSONL}
GPUs: ${GPU_IDS}
World size: ${NPROC_PER_NODE}
DeepSpeed: ZeRO-2
Per-device batch: ${PER_DEVICE_BATCH_SIZE}
Gradient accumulation: ${GRAD_ACCUM_STEPS}
Effective global batch: ${GLOBAL_BATCH}
Epochs: ${NUM_EPOCHS}
Optimizer: adamw_torch_fused
Gradient checkpointing: True
Dataloader workers: ${DATALOADER_NUM_WORKERS}
Persistent workers: True
Prefetch factor: ${DATALOADER_PREFETCH_FACTOR}
Image pixels: ${MIN_PIXELS}..${MAX_PIXELS}
LR: 3e-6
Scheduler: cosine
Warmup ratio: 0.05
Eval milestones: 25%, 50%, 75%, 100% of actual optimizer steps
Resume checkpoints retained: 1
EOF

printf '\nGroundCUA Qwen3-VL-8B full SFT\n'
cat "${RUN_INFO}"
printf 'Output: %s\n' "${OUTPUT_DIR}"
printf 'Snapshots: %s\n' "${EVAL_SNAPSHOT_DIR}"
printf 'Logs: %s\n\n' "${LOG_DIR}"

# Record physical GPU memory/utilization throughout the multi-day run.
nvidia-smi \
  --id=${GPU_IDS} \
  --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader,nounits \
  -l "${GPU_MONITOR_INTERVAL}" > "${GPU_LOG}" 2>&1 &
monitor_pid=$!

set +e
CUDA_VISIBLE_DEVICES=${GPU_IDS} \
torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  qwenvl/train/train_qwen.py \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use groundcua_sft \
  --data_flatten True \
  --data_packing False \
  --decode_unused_text False \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs ${NUM_EPOCHS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --max_pixels ${MAX_PIXELS} \
  --min_pixels ${MIN_PIXELS} \
  --eval_strategy no \
  --save_strategy no \
  --save_total_limit 1 \
  --save_eval_snapshots True \
  --eval_snapshot_dir "${EVAL_SNAPSHOT_DIR}" \
  --learning_rate 3e-6 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --optim adamw_torch_fused \
  --logging_steps 1 \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor ${DATALOADER_PREFETCH_FACTOR} \
  --dataloader_pin_memory True \
  --report_to none \
  2>&1 | tee "${TRAIN_LOG}"
rc=${PIPESTATUS[0]}
set -e

cleanup_monitor
monitor_pid=""

if [[ ${rc} -ne 0 ]]; then
  exit "${rc}"
fi

printf '\nTraining completed successfully.\n'
printf 'Final HF model: %s\n' "${OUTPUT_DIR}"
printf 'Evaluation snapshots:\n'
find "${EVAL_SNAPSHOT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '  %f\n' 2>/dev/null | sort || true
printf 'Latest resumable checkpoint:\n'
find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '  %f\n' 2>/dev/null | sort -V || true
