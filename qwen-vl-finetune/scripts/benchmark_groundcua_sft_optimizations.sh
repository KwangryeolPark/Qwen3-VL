#!/usr/bin/env bash
set -uo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
SFT_ROOT=${ROOT}/qwen3-vl-src/qwen-vl-finetune
SMOKE_SCRIPT=${SFT_ROOT}/scripts/sft_qwen3_8b_groundcua_smoke.sh
GPU_IDS=${GPU_IDS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MAX_STEPS=${MAX_STEPS:-12}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-1}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-10}
KEEP_OUTPUTS=${KEEP_OUTPUTS:-0}

STAMP=$(date +%Y%m%d_%H%M%S)
BENCH_DIR=${BENCH_DIR:-${SFT_ROOT}/benchmark_logs/groundcua_sft_optim_${STAMP}}
SUMMARY_CSV=${BENCH_DIR}/summary.csv
mkdir -p "${BENCH_DIR}"
cd "${SFT_ROOT}"
chmod +x "${SMOKE_SCRIPT}" 2>/dev/null || true

notify_email() {
  local rc=${1:-0}

  # Optional notification: silently skip on systems without NOTIFY_EMAIL/mail.
  if [[ -z "${NOTIFY_EMAIL:-}" ]]; then
    return 0
  fi
  if ! command -v mail >/dev/null 2>&1; then
    echo "NOTIFY_EMAIL is set, but 'mail' is not available; skipping notification." >&2
    return 0
  fi

  local host status subject
  host=$(hostname)
  if [[ ${rc} -eq 0 ]]; then
    status="COMPLETED"
  else
    status="FAILED (exit=${rc})"
  fi
  subject="[GroundCUA SFT] Optimization sweep ${status} - ${host}"

  {
    echo "GroundCUA SFT optimization sweep finished."
    echo
    echo "Host: ${host}"
    echo "Status: ${status}"
    echo "Finished: $(date --iso-8601=seconds)"
    echo "Benchmark directory: ${BENCH_DIR}"
    echo "Summary CSV: ${SUMMARY_CSV}"
    echo
    if [[ -f "${SUMMARY_CSV}" ]]; then
      echo "==================== SUMMARY ===================="
      if command -v column >/dev/null 2>&1; then
        column -s, -t < "${SUMMARY_CSV}"
      else
        cat "${SUMMARY_CSV}"
      fi
    fi
  } | mail -s "${subject}" "${NOTIFY_EMAIL}" || true
}

on_exit() {
  local rc=$?
  trap - EXIT
  notify_email "${rc}"
  exit "${rc}"
}
trap on_exit EXIT

# case|zero_cfg|bs|ga|decode|workers|persistent|prefetch|optim|grad_ckpt
DEFAULT_CASES=$(cat <<'EOF'
legacy_z3_bs2|./scripts/zero3.json|2|16|True|8|False|2|adamw_torch|True
nodecode_z3_bs2|./scripts/zero3.json|2|16|False|8|False|2|adamw_torch|True
loader_z3_bs2|./scripts/zero3.json|2|16|False|16|True|4|adamw_torch|True
fused_z3_bs2|./scripts/zero3.json|2|16|False|16|True|4|adamw_torch_fused|True
zero2_bs1|./scripts/zero2.json|1|32|False|16|True|4|adamw_torch_fused|True
zero2_bs2|./scripts/zero2.json|2|16|False|16|True|4|adamw_torch_fused|True
nogc_z3_bs1|./scripts/zero3.json|1|32|False|16|True|4|adamw_torch_fused|False
nogc_z3_bs2|./scripts/zero3.json|2|16|False|16|True|4|adamw_torch_fused|False
EOF
)
CASES=${CASES:-${DEFAULT_CASES}}

cat > "${SUMMARY_CSV}" <<'EOF'
case,zero_stage,per_device_batch,grad_accum,global_batch,decode_unused_text,dataloader_workers,persistent_workers,prefetch_factor,optimizer,gradient_checkpointing,max_steps,status,exit_code,wall_seconds,train_runtime_seconds,train_steps_per_second,seconds_per_step,peak_gpu_memory_mib,avg_gpu_util_pct,train_log,gpu_log
EOF

cleanup_monitor() {
  local pid=${1:-}
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

summarize_gpu_log() {
  python - "$1" <<'PY'
import csv, sys
peak = 0.0
utils = []
try:
    with open(sys.argv[1], newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            try:
                mem = float(row[1].strip())
                util = float(row[3].strip())
            except ValueError:
                continue
            peak = max(peak, mem)
            utils.append(util)
except FileNotFoundError:
    pass
print(f"{peak:.0f},{(sum(utils)/len(utils) if utils else 0.0):.2f}")
PY
}

extract_train_metrics() {
  python - "$1" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read()
def last(pattern):
    vals = re.findall(pattern, text)
    return vals[-1] if vals else ""
runtime = last(r"['\"]train_runtime['\"]\s*:\s*([0-9.eE+-]+)")
stepsps = last(r"['\"]train_steps_per_second['\"]\s*:\s*([0-9.eE+-]+)")
print(f"{runtime},{stepsps}")
PY
}

zero_stage_from_cfg() {
  if [[ "$1" == *zero2* ]]; then echo 2; else echo 3; fi
}

printf '\nGroundCUA SFT optimization sweep\n'
printf '  GPUs:             %s\n' "${GPU_IDS}"
printf '  max steps/run:    %s\n' "${MAX_STEPS}"
printf '  monitor interval: %ss\n' "${MONITOR_INTERVAL}"
printf '  logs:             %s\n\n' "${BENCH_DIR}"

while IFS='|' read -r name zero_cfg bs ga decode workers persistent prefetch optim grad_ckpt; do
  [[ -z "${name}" ]] && continue
  global_batch=$((NPROC_PER_NODE * bs * ga))
  zero_stage=$(zero_stage_from_cfg "${zero_cfg}")
  train_log=${BENCH_DIR}/${name}.train.log
  gpu_log=${BENCH_DIR}/${name}.gpu.csv
  output_dir=${BENCH_DIR}/${name}.output

  printf '\n============================================================\n'
  printf 'Case: %s\n' "${name}"
  printf '  ZeRO-%s | bs=%s | ga=%s | global=%s | optim=%s | gc=%s\n' \
    "${zero_stage}" "${bs}" "${ga}" "${global_batch}" "${optim}" "${grad_ckpt}"
  printf '  decode=%s | workers=%s | persistent=%s | prefetch=%s\n' \
    "${decode}" "${workers}" "${persistent}" "${prefetch}"
  printf '============================================================\n'

  nvidia-smi \
    --id=${GPU_IDS} \
    --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    -l "${MONITOR_INTERVAL}" > "${gpu_log}" 2>&1 &
  monitor_pid=$!

  start_epoch=$(date +%s)
  set +e
  CUDA_VISIBLE_DEVICES=${GPU_IDS} \
  NPROC_PER_NODE=${NPROC_PER_NODE} \
  PER_DEVICE_BATCH_SIZE=${bs} \
  GRAD_ACCUM_STEPS=${ga} \
  MAX_STEPS=${MAX_STEPS} \
  SAVE_STRATEGY=no \
  SKIP_FINAL_SAVE=True \
  OUTPUT_DIR=${output_dir} \
  DEEPSPEED_CONFIG=${zero_cfg} \
  OPTIM=${optim} \
  GRADIENT_CHECKPOINTING=${grad_ckpt} \
  DATALOADER_NUM_WORKERS=${workers} \
  DATALOADER_PERSISTENT_WORKERS=${persistent} \
  DATALOADER_PREFETCH_FACTOR=${prefetch} \
  DATALOADER_PIN_MEMORY=True \
  DECODE_UNUSED_TEXT=${decode} \
  bash "${SMOKE_SCRIPT}" 2>&1 | tee "${train_log}"
  rc=${PIPESTATUS[0]}
  set -e
  end_epoch=$(date +%s)
  wall_seconds=$((end_epoch - start_epoch))
  cleanup_monitor "${monitor_pid}"

  status="SUCCESS"
  if [[ ${rc} -ne 0 ]]; then
    status="FAILED"
    if grep -Eqi 'CUDA out of memory|out of memory|torch\.OutOfMemoryError' "${train_log}"; then
      status="OOM"
    fi
  fi

  IFS=, read -r peak_mem avg_util <<< "$(summarize_gpu_log "${gpu_log}")"
  IFS=, read -r train_runtime train_stepsps <<< "$(extract_train_metrics "${train_log}")"

  sec_per_step=""
  if [[ -n "${train_runtime}" ]]; then
    sec_per_step=$(python - "${train_runtime}" "${MAX_STEPS}" <<'PY'
import sys
print(f"{float(sys.argv[1])/int(sys.argv[2]):.3f}")
PY
)
  fi

  printf '\nResult %s\n' "${name}"
  printf '  status:              %s (exit=%s)\n' "${status}" "${rc}"
  printf '  wall time:           %ss\n' "${wall_seconds}"
  printf '  trainer runtime:     %ss\n' "${train_runtime:-N/A}"
  printf '  sec/optimizer step:  %s\n' "${sec_per_step:-N/A}"
  printf '  trainer steps/sec:   %s\n' "${train_stepsps:-N/A}"
  printf '  peak GPU memory:     %s MiB\n' "${peak_mem}"
  printf '  avg GPU utilization: %s%%\n' "${avg_util}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${name}" "${zero_stage}" "${bs}" "${ga}" "${global_batch}" "${decode}" \
    "${workers}" "${persistent}" "${prefetch}" "${optim}" "${grad_ckpt}" "${MAX_STEPS}" \
    "${status}" "${rc}" "${wall_seconds}" "${train_runtime}" "${train_stepsps}" \
    "${sec_per_step}" "${peak_mem}" "${avg_util}" "${train_log}" "${gpu_log}" >> "${SUMMARY_CSV}"

  if [[ "${KEEP_OUTPUTS}" != "1" ]]; then
    rm -rf "${output_dir}"
  fi
  sleep "${COOLDOWN_SECONDS}"
done <<< "${CASES}"

printf '\n============================================================\n'
printf 'Optimization sweep complete. Summary:\n'
column -s, -t < "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
printf '\nSummary CSV: %s\n' "${SUMMARY_CSV}"
printf 'All logs:   %s\n' "${BENCH_DIR}"
