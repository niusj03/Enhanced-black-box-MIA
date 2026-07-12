#!/usr/bin/env bash
set -euo pipefail

# Sweep WPMIA tau/gamma over complete WikiMIA caches for the four paper models.
# This reuses records.jsonl and does not intentionally rerun target-LLM sampling.
#
# Usage:
#   nohup bash scripts/run_wpmia_wikimia_tau_gamma_sweep.sh "0 1 2 3 4 5 6 7" \
#     > logs/wpmia/wikimia_wpmia_sweep.nohup.log 2>&1 &
#
# Useful overrides:
#   WIKIMIA_MODELS="EleutherAI/pythia-6.9b facebook/opt-6.7b huggyllama/llama-13b EleutherAI/gpt-neox-20b"
#   WIKIMIA_LENGTHS="32 64 128"
#   WPMIA_TAUS="0.03 0.05 0.1 0.2 0.5"
#   WPMIA_GAMMAS="0 0.25 0.5 0.75 1.0"
#   CSV_ROOT="logs/wpmia"
#   RUN_ROOT="logs/wpmia/$(date +%Y%m%d_%H%M%S)_wikimia_wpmia"
#   RESET_CSV=1
#   CONTINUE_ON_ERROR=1
#   DRY_RUN=1
#   SIMMIA_BIN=/path/to/simmia.benchmark

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

DATA="swj0419/WikiMIA"
CACHE_ROOT="${CACHE_ROOT:-simmia_out/WikiMIA}"
MODELS="${WIKIMIA_MODELS:-EleutherAI/pythia-6.9b facebook/opt-6.7b huggyllama/llama-13b EleutherAI/gpt-neox-20b}"
LENGTHS="${WIKIMIA_LENGTHS:-32 64 128}"
TAUS="${WPMIA_TAUS:-0.03 0.05 0.1 0.2 0.5}"
GAMMAS="${WPMIA_GAMMAS:-0 0.25 0.5 0.75 1.0}"
NUM_SHOTS="${NUM_SHOTS:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-simmia_out}"
DRY_RUN="${DRY_RUN:-0}"
RESET_CSV="${RESET_CSV:-0}"

if [[ -n "${SIMMIA_BIN:-}" ]]; then
  SIMMIA_CMD=("$SIMMIA_BIN")
elif command -v simmia.benchmark >/dev/null 2>&1; then
  SIMMIA_CMD=("simmia.benchmark")
elif [[ -x "$HOME/miniconda3/envs/simmia/bin/simmia.benchmark" ]]; then
  SIMMIA_CMD=("$HOME/miniconda3/envs/simmia/bin/simmia.benchmark")
else
  SIMMIA_CMD=("simmia.benchmark")
fi

RUN_ROOT="${RUN_ROOT:-logs/wpmia/$(date +%Y%m%d_%H%M%S)_wikimia_wpmia}"
CSV_ROOT="${CSV_ROOT:-logs/wpmia}"
mkdir -p "$RUN_ROOT" "$CSV_ROOT"
SUMMARY_LOG="$RUN_ROOT/summary.log"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-simmia}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$MPLCONFIGDIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$SUMMARY_LOG"
}

slugify() {
  local s="$1"
  s="${s#api:}"
  s="${s//\//__}"
  s="${s//[^A-Za-z0-9._-]/_}"
  printf '%s' "$s"
}

format_param_slug() {
  local s="$1"
  s="${s//./p}"
  s="${s//-/_neg_}"
  printf '%s' "$s"
}

csv_for_model() {
  local model_base="$1"
  case "$model_base" in
    pythia-6.9b) printf '%s/wikimia_pythia_wpmia_sweep.csv' "$CSV_ROOT" ;;
    opt-6.7b) printf '%s/wikimia_opt67b_wpmia_sweep.csv' "$CSV_ROOT" ;;
    llama-13b) printf '%s/wikimia_llama13b_wpmia_sweep.csv' "$CSV_ROOT" ;;
    gpt-neox-20b) printf '%s/wikimia_gptneox20b_wpmia_sweep.csv' "$CSV_ROOT" ;;
    *) printf '%s/wikimia_%s_wpmia_sweep.csv' "$CSV_ROOT" "$(slugify "$model_base")" ;;
  esac
}

append_csv_row() {
  local csv_path="$1"
  local status="$2"
  local model="$3"
  local length="$4"
  local tau="$5"
  local gamma="$6"
  local result_name="$7"
  local output_dir="$8"
  local roc_path="$9"
  local log_file="${10}"

  python - "$csv_path" "$status" "$model" "$DATA" "$length" "$tau" "$gamma" \
    "$NUM_SHOTS" "$result_name" "$output_dir" "$roc_path" "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    model,
    data,
    length,
    tau,
    gamma,
    num_shots,
    result_name,
    output_dir,
    roc_path,
    log_file,
) = sys.argv[1:]

metrics = {
    "auc_pct": "",
    "accuracy_pct": "",
    "tpr1_fpr_pct": "",
    "tpr5_fpr_pct": "",
    "tpr10_fpr_pct": "",
}
pattern = re.compile(
    r"AUC (?P<auc>[0-9.]+), Accuracy (?P<acc>[0-9.]+), "
    r"TPR@1%FPR of (?P<tpr1>[0-9.]+), TPR@5%FPR of (?P<tpr5>[0-9.]+), "
    r"TPR@10%FPR of (?P<tpr10>[0-9.]+)"
)

if os.path.exists(log_file):
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                metrics = {
                    "auc_pct": match.group("auc"),
                    "accuracy_pct": match.group("acc"),
                    "tpr1_fpr_pct": match.group("tpr1"),
                    "tpr5_fpr_pct": match.group("tpr5"),
                    "tpr10_fpr_pct": match.group("tpr10"),
                }

row = {
    "status": status,
    "model": model,
    "data": data,
    "sub_dataset": f"WikiMIA_length{length}",
    "length": length,
    "tau": tau,
    "gamma": gamma,
    "num_shots": num_shots,
    "auc_pct": metrics["auc_pct"],
    "accuracy_pct": metrics["accuracy_pct"],
    "tpr1_fpr_pct": metrics["tpr1_fpr_pct"],
    "tpr5_fpr_pct": metrics["tpr5_fpr_pct"],
    "tpr10_fpr_pct": metrics["tpr10_fpr_pct"],
    "result_name": result_name,
    "output_dir": output_dir,
    "roc_path": roc_path,
    "log_file": log_file,
}

fieldnames = [
    "status",
    "model",
    "data",
    "sub_dataset",
    "length",
    "tau",
    "gamma",
    "num_shots",
    "auc_pct",
    "accuracy_pct",
    "tpr1_fpr_pct",
    "tpr5_fpr_pct",
    "tpr10_fpr_pct",
    "result_name",
    "output_dir",
    "roc_path",
    "log_file",
]

write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

check_cache_complete() {
  local cache_dir="$1"
  local records="$cache_dir/records.jsonl"
  local full="$cache_dir/full_dataset.jsonl"

  python - "$records" "$full" <<'PY'
import json
import os
import sys

records_path, full_path = sys.argv[1:]
required = {
    "label_results",
    "sample_results",
    "nonmember_prefix_sample_results",
    "member_prefix_sample_results",
}

if not os.path.exists(records_path):
    print(f"missing records.jsonl: {records_path}")
    sys.exit(2)
if not os.path.exists(full_path):
    print(f"missing full_dataset.jsonl: {full_path}")
    sys.exit(2)

with open(records_path, "r", encoding="utf-8") as f:
    first = f.readline()
if not first.strip():
    print(f"empty records.jsonl: {records_path}")
    sys.exit(2)

record = json.loads(first)
missing = sorted(required - set(record))
if missing:
    print("missing required record fields: " + ",".join(missing))
    sys.exit(2)

with open(records_path, "rb") as f:
    records_count = sum(1 for _ in f)
with open(full_path, "rb") as f:
    full_count = sum(1 for _ in f)
if records_count != full_count:
    print(f"incomplete cache: records={records_count}, full_dataset={full_count}")
    sys.exit(2)
PY
}

run_one_setting() {
  local model="$1"
  local model_base="$2"
  local length="$3"
  local tau="$4"
  local gamma="$5"
  local csv_path="$6"
  local output_dir="$7"
  local sub_dataset="WikiMIA_length${length}"
  local tau_slug
  local gamma_slug
  local result_name
  local roc_path
  local log_file
  local status

  tau_slug="$(format_param_slug "$tau")"
  gamma_slug="$(format_param_slug "$gamma")"
  result_name="wpmia_len${length}_tau_${tau_slug}_gamma_${gamma_slug}"
  roc_path="${output_dir}/${result_name}_roc_tpr_at_5_fpr.png"
  log_file="${RUN_ROOT}/$(slugify "${model_base}_${sub_dataset}_${result_name}").log"

  cmd=(
    "${SIMMIA_CMD[@]}"
    --model_name_or_path "$model"
    --sampling relative_word_by_word
    --postprocess process_wpmia_word_data
    --inference wpmia_score
    --output_dir "$OUTPUT_DIR"
    --result_name "$result_name"
    --num_samples "$NUM_SAMPLES"
    --data "$DATA"
    --sub_dataset "$sub_dataset"
    --num_shots "$NUM_SHOTS"
    --prefix_ratio 0.0
    --top_k 20
    --wpmia_tau "$tau"
    --wpmia_gamma "$gamma"
  )

  if [[ -n "${GPU_IDS:-}" ]]; then
    local gpu_ids_normalized="${GPU_IDS//,/ }"
    local -a gpu_id_arr=()
    read -r -a gpu_id_arr <<< "$gpu_ids_normalized"
    cmd+=(--gpu_ids "${gpu_id_arr[@]}")
  fi
  if [[ -n "${CONCURRENCY:-}" ]]; then
    cmd+=(--concurrency "$CONCURRENCY")
  fi
  cmd+=("${EXTRA_ARGS[@]}")

  log "START model=${model} length=${length} tau=${tau} gamma=${gamma}"
  {
    printf 'COMMAND:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
  } > "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    append_csv_row "$csv_path" "dry_run" "$model" "$length" "$tau" "$gamma" \
      "$result_name" "$output_dir" "$roc_path" "$log_file"
    log "DONE  model=${model} length=${length} tau=${tau} gamma=${gamma} status=dry_run"
    return 0
  fi

  status="ok"
  if ! "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    status="failed"
  fi

  append_csv_row "$csv_path" "$status" "$model" "$length" "$tau" "$gamma" \
    "$result_name" "$output_dir" "$roc_path" "$log_file"

  log "DONE  model=${model} length=${length} tau=${tau} gamma=${gamma} status=${status}"
  if [[ "$status" != "ok" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
}

log "Run root: $RUN_ROOT"
log "CSV root: $CSV_ROOT"
log "WikiMIA cache root: $CACHE_ROOT"
log "Models: $MODELS"
log "Lengths: $LENGTHS"
log "Tau values: $TAUS"
log "Gamma values: $GAMMAS"

if [[ ! -d "$CACHE_ROOT" ]]; then
  log "ERROR missing WikiMIA cache root: $CACHE_ROOT"
  exit 2
fi

declare -A reset_done=()
total_caches=0

for model in $MODELS; do
  model_base="${model##*/}"
  csv_path="$(csv_for_model "$model_base")"

  if [[ "$RESET_CSV" == "1" && -z "${reset_done[$csv_path]:-}" ]]; then
    : > "$csv_path"
    reset_done[$csv_path]=1
  fi

  log "MODEL $model"
  log "CSV   $csv_path"

  for length in $LENGTHS; do
    sub_dataset="WikiMIA_length${length}"
    cache_dir="${CACHE_ROOT}/${sub_dataset}/${model_base}/${NUM_SHOTS}"
    records_path="${cache_dir}/records.jsonl"

    if [[ ! -f "$records_path" ]]; then
      log "SKIP missing cache: $cache_dir"
      continue
    fi
    if ! cache_check_output="$(check_cache_complete "$cache_dir" 2>&1)"; then
      log "SKIP incomplete cache: $cache_dir"
      log "$cache_check_output"
      continue
    fi

    total_caches=$((total_caches + 1))
    log "CACHE $cache_dir"

    for tau in $TAUS; do
      for gamma in $GAMMAS; do
        run_one_setting "$model" "$model_base" "$length" "$tau" "$gamma" \
          "$csv_path" "$cache_dir"
      done
    done
  done
done

log "Finished. Swept $total_caches complete WikiMIA caches."
