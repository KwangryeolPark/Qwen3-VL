#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
MODEL_PATH=${MODEL_PATH:-${ROOT}/Qwen3-VL-8B-Instruct}
SFT_ROOT=${ROOT}/qwen3-vl-src/qwen-vl-finetune
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/qwen3-vl-src/qwen-vl-finetune/output/groundcua-qwen3-vl-8b-sft}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

cd "${SFT_ROOT}"

export GROUND_CUA_SFT_JSONL=${GROUND_CUA_SFT_JSONL:-${ROOT}/datasets/groundcua/sft/qwen3vl_train.jsonl}

# GroundCUA paper-inspired recipe adapted to 4x H100:
#   - full model SFT
#   - 2 epochs
#   - LR 3e-6, cosine, warmup 0.05
#   - effective global batch 128 = 4 GPUs * 1 sample/GPU * grad_accum 32
# We intentionally keep the image pixel budget aligned with the later EasyR1 GRPO
# stage (262144..2359296) for a clean SFT -> RL handoff.  Override MAX_PIXELS if
# reproducing GroundCUA's larger original image budget is desired.
MAX_PIXELS=${MAX_PIXELS:-2359296}
MIN_PIXELS=${MIN_PIXELS:-262144}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-32}
SAVE_STEPS=${SAVE_STEPS:-500}

torchrun \
  --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  qwenvl/train/train_qwen.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use groundcua_sft \
  --data_flatten True \
  --data_packing False \
  --tune_mm_vision True \
  --tune_mm_mlp True \
  --tune_mm_llm True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --max_pixels ${MAX_PIXELS} \
  --min_pixels ${MIN_PIXELS} \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit 3 \
  --learning_rate 3e-6 \
  --weight_decay 0 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --model_max_length 8192 \
  --gradient_checkpointing True \
  --dataloader_num_workers 16 \
  --report_to none
