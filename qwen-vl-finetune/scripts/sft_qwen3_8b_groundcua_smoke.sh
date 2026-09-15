#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
MODEL_PATH=${MODEL_PATH:-${ROOT}/Qwen3-VL-8B-Instruct}
SFT_ROOT=${ROOT}/qwen3-vl-src/qwen-vl-finetune
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# Tunable smoke/throughput-test knobs.
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
MAX_STEPS=${MAX_STEPS:-100}
MAX_PIXELS=${MAX_PIXELS:-2359296}
MIN_PIXELS=${MIN_PIXELS:-262144}
SAVE_STRATEGY=${SAVE_STRATEGY:-steps}
SAVE_STEPS=${SAVE_STEPS:-${MAX_STEPS}}
OUTPUT_DIR=${OUTPUT_DIR:-${SFT_ROOT}/output/groundcua-qwen3-vl-8b-sft-smoke-bs${PER_DEVICE_BATCH_SIZE}-ga${GRAD_ACCUM_STEPS}}

cd "${SFT_ROOT}"

# Smoke data: generated separately with tools/prepare_groundcua_sft.py --limit 1000.
export GROUND_CUA_SFT_SMOKE_JSONL=${GROUND_CUA_SFT_SMOKE_JSONL:-${ROOT}/datasets/groundcua/sft/qwen3vl_train_smoke.jsonl}

GLOBAL_BATCH=$((NPROC_PER_NODE * PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS))
echo "GroundCUA SFT smoke/throughput test"
echo "  GPUs:                  ${NPROC_PER_NODE}"
echo "  per-device batch:      ${PER_DEVICE_BATCH_SIZE}"
echo "  grad accumulation:     ${GRAD_ACCUM_STEPS}"
echo "  effective global batch:${GLOBAL_BATCH}"
echo "  max steps:             ${MAX_STEPS}"
echo "  output:                ${OUTPUT_DIR}"

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
  --max_steps ${MAX_STEPS} \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --max_pixels ${MAX_PIXELS} \
  --min_pixels ${MIN_PIXELS} \
  --save_strategy ${SAVE_STRATEGY} \
  --save_steps ${SAVE_STEPS} \
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
