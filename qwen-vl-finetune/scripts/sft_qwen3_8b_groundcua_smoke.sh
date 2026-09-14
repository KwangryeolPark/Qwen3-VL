#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
MODEL_PATH=${MODEL_PATH:-${ROOT}/Qwen3-VL-8B-Instruct}
SFT_ROOT=${ROOT}/qwen3-vl-src/qwen-vl-finetune
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/qwen3-vl-src/qwen-vl-finetune/output/groundcua-qwen3-vl-8b-sft-smoke}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

cd "${SFT_ROOT}"

# Smoke data: generated separately with tools/prepare_groundcua_sft.py --limit 1000.
export GROUND_CUA_SFT_SMOKE_JSONL=${GROUND_CUA_SFT_SMOKE_JSONL:-${ROOT}/datasets/groundcua/sft/qwen3vl_train_smoke.jsonl}

# Keep this small enough to finish quickly while exercising the real full-parameter path.
# Global batch = 4 GPUs * per-device 1 * grad-accum 4 = 16.
# max_steps=100 gives a useful throughput/memory measurement before the multi-day run.
torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  qwenvl/train/train_qwen.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use groundcua_sft_smoke \
  --data_flatten True \
  --data_packing False \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_pixels 2359296 \
  --min_pixels 262144 \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 1 \
  --learning_rate 3e-6 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --report_to none
