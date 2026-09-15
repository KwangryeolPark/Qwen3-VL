#!/usr/bin/env bash
set -uo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
SFT_ROOT=${ROOT}/qwen3-vl-src/qwen-vl-finetune
SMOKE_SCRIPT=${SFT_ROOT}/scripts/sft_qwen3_8b_groundcua_smoke.sh
GPU_IDS=${GPU_IDS:-0,1,2,3}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MAX_STEPS=${MAX_STEPS:-20}
MONITOR_INTERVAL=${MONITOR_INTERVAL:-1}
CONFIGS=${CONFIGS:-"1:32 2:16 4:8 8:4"}
STOP_ON_OOM=${STOP_ON_OOM:-1}
KEEP_OUTPUTS=${KEEP_OUTPUTS:-0}
COOLDOWN_SECONDS=${COOLDOWN_SECONDS:-10}

STAMP=$(date +%Y%m%d_%H%M%S)
BENCH_DIR=${BENCH_DIR:-${SFT_ROOT}/benchmark_logs/groundcua_sft_${STAMP}}
SUMMARY_CSV=${BENCH_DIR}/summary.csv

mkdir -p "${BENCH_DIR}"
cd "${SFT_ROOT}"

if [[ ! -x "${SMOKE_SCRIPT}" ]]; then
  chmod +x "${SMOKE_SCRIPT}"
fi

cat > "${SUMMARY_CSV}" <<'EOF'
per_device_batch,grad_accum,global_batch,max_steps,status,exit_code,wall_seconds,train_runtime_seconds,train_steps_per_second,peak_gpu_memory_mib,avg_gpu_util_pct,train_log,gpu_log
EOF

cleanup_monitor() {
  local pid=${1:-}
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

summarize_gpu_log() {
  local gpu_log=$1
  python - "${gpu_log}" <<'PY'
import csv
import sys

path = sys.argv[1]
peak = 0.0
utils = []
try:
    with open(path, newline="") as f:
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
avg = sum(utils) / len(utils) if utils else 0.0
print(f"{peak:.0f},{avg:.2f}")
PY
}

extract_train_metrics() {
  local train_log=$1
  python - "${train_log}" <<'PY'
import re
import sys

text = open(sys.argv[1], errors="replace").read()

def last(pattern):
    vals = re.findall(pattern, text)
    return vals[-1] if vals else ""

runtime = last(r"['\"]train_runtime['\"]\s*:\s*([0-9.eE+-]+)")
stepsps = last(r"['\"]train_steps_per_second['\"]\s*:\s*([0-9.eE+-]+)")
print(f"{runtime},{stepsps}")
PY
}

printf '\nGroundCUA SFT throughput sweep\n'
printf '  configs:          %s\n' "${CONFIGS}"
printf '  GPUs:             %s\n' "${GPU_IDS}"
printf '  max steps/run:    %s\n' "${MAX_STEPS}"
printf '  monitor interval: %ss\n' "${MONITOR_INTERVAL}"
printf '  logs:             %s\n\n' "${BENCH_DIR}"

for cfg in ${CONFIGS}; do
  bs=${cfg%%:*}
  ga=${cfg##*:}
  global_batch=$((NPROC_PER_NODE * bs * ga))
  tag="bs${bs}_ga${ga}"
  train_log=${BENCH_DIR}/${tag}.train.log
  gpu_log=${BENCH_DIR}/${tag}.gpu.csv
  output_dir=${BENCH_DIR}/${tag}.output

  printf '\n============================================================\n'
  printf 'Testing %s: per-device batch=%s, grad_accum=%s, global batch=%s\n' \
    "${tag}" "${bs}" "${ga}" "${global_batch}"
  printf '============================================================\n'

  # Sample physical GPU memory and utilization once per second.
  nvidia-smi \
    --id=${GPU_IDS} \
    --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits \
    -l "${MONITOR_INTERVAL}" > "${gpu_log}" 2>&1 &
  monitor_pid=$!

  start_epoch=$(date +%s)
  start_iso=$(date --iso-8601=seconds)

  set +e
  CUDA_VISIBLE_DEVICES=${GPU_IDS} \
  NPROC_PER_NODE=${NPROC_PER_NODE} \
  PER_DEVICE_BATCH_SIZE=${bs} \
  GRAD_ACCUM_STEPS=${ga} \
  MAX_STEPS=${MAX_STEPS} \
  SAVE_STRATEGY=no \
  OUTPUT_DIR=${output_dir} \
  bash "${SMOKE_SCRIPT}" 2>&1 | tee "${train_log}"
  rc=${PIPESTATUS[0]}
  set -e

  end_epoch=$(date +%s)
  end_iso=$(date --iso-8601=seconds)
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

  printf '\nResult %s\n' "${tag}"
  printf '  status:             %s (exit=%s)\n' "${status}" "${rc}"
  printf '  started:            %s\n' "${start_iso}"
  printf '  finished:           %s\n' "${end_iso}"
  printf '  wall time:          %ss\n' "${wall_seconds}"
  printf '  trainer runtime:    %ss\n' "${train_runtime:-N/A}"
  printf '  trainer steps/sec:  %s\n' "${train_stepsps:-N/A}"
  printf '  peak GPU memory:    %s MiB\n' "${peak_mem}"
  printf '  avg GPU utilization:%s%%\n' "${avg_util}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${bs}" "${ga}" "${global_batch}" "${MAX_STEPS}" "${status}" "${rc}" \
    "${wall_seconds}" "${train_runtime}" "${train_stepsps}" "${peak_mem}" "${avg_util}" \
    "${train_log}" "${gpu_log}" >> "${SUMMARY_CSV}"

  if [[ "${KEEP_OUTPUTS}" != "1" ]]; then
    rm -rf "${output_dir}"
  fi

  # Let CUDA contexts/processes disappear before the next configuration.
  sleep "${COOLDOWN_SECONDS}"

  if [[ "${status}" == "OOM" && "${STOP_ON_OOM}" == "1" ]]; then
    echo "Stopping sweep after OOM; larger per-device batches are unlikely to fit."
    break
  fi
done

printf '\n============================================================\n'
printf 'Sweep complete. Summary:\n'
column -s, -t < "${SUMMARY_CSV}" 2>/dev/null || cat "${SUMMARY_CSV}"
printf '\nSummary CSV: %s\n' "${SUMMARY_CSV}"
printf 'All logs:   %s\n' "${BENCH_DIR}"
