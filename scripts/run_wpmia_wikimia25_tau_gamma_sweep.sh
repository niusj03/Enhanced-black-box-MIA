#!/usr/bin/env bash
set -euo pipefail

# Sweep WPMIA tau/gamma over complete WikiMIA-25 caches under simmia_out/WikiMIA-25.
# This reuses records.jsonl and does not intentionally rerun target-LLM sampling.
#
# Usage:
#   nohup bash scripts/run_wpmia_wikimia25_tau_gamma_sweep.sh "0 1 2 3 4 5 6 7" \
#     > logs/wpmia/wikimia25_wpmia_sweep.nohup.log 2>&1 &
#
# Useful overrides:
#   CACHE_ROOT="simmia_out/WikiMIA-25"
#   WPMIA_TAUS="0.03 0.05 0.1 0.2 0.5"
#   WPMIA_GAMMAS="0 0.25 0.5 0.75 1.0"
#   CSV_ROOT="logs/wpmia"
#   RUN_ROOT="logs/wpmia/$(date +%Y%m%d_%H%M%S)_wikimia25_wpmia"
#   CONTINUE_ON_ERROR=1
#   DRY_RUN=1
#   SIMMIA_BIN=/path/to/simmia.benchmark

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

CACHE_ROOT="${CACHE_ROOT:-simmia_out/WikiMIA-25}"
TAUS="${WPMIA_TAUS:-0.03 0.05 0.1 0.2 0.5}"
GAMMAS="${WPMIA_GAMMAS:-0 0.25 0.5 0.75 1.0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-simmia_out}"
DRY_RUN="${DRY_RUN:-0}"
DATA="SimMIA/WikiMIA-25"
NUM_SAMPLES="10"

if [[ -n "${SIMMIA_BIN:-}" ]]; then
  SIMMIA_CMD=("$SIMMIA_BIN")
elif command -v simmia.benchmark >/dev/null 2>&1; then
  SIMMIA_CMD=("simmia.benchmark")
elif [[ -x "$HOME/miniconda3/envs/simmia/bin/simmia.benchmark" ]]; then
  SIMMIA_CMD=("$HOME/miniconda3/envs/simmia/bin/simmia.benchmark")
else
  SIMMIA_CMD=("simmia.benchmark")
fi

RUN_ROOT="${RUN_ROOT:-logs/wpmia/$(date +%Y%m%d_%H%M%S)_wikimia25_wpmia}"
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
  local model="$1"
  printf '%s/wikimia25_%s_wpmia_sweep.csv' "$CSV_ROOT" "$(slugify "$model")"
}

append_csv_row() {
  local csv_path="$1"
  local status="$2"
  local sub_dataset="$3"
  local model="$4"
  local num_shots="$5"
  local tau="$6"
  local gamma="$7"
  local result_name="$8"
  local output_dir="$9"
  local roc_path="${10}"
  local log_file="${11}"

  python - "$csv_path" "$status" "$DATA" "$sub_dataset" "$model" \
    "$num_shots" "$tau" "$gamma" "$result_name" "$output_dir" "$roc_path" \
    "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    data,
    sub_dataset,
    model,
    num_shots,
    tau,
    gamma,
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
    "benchmark": "wikimia25",
    "model": model,
    "data": data,
    "split": "",
    "sub_dataset": sub_dataset,
    "num_shots": num_shots,
    "tau": tau,
    "gamma": gamma,
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
    "benchmark",
    "model",
    "data",
    "split",
    "sub_dataset",
    "num_shots",
    "tau",
    "gamma",
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

parse_wikimia25_cache_path() {
  local records="$1"
  local rel
  rel="${records#simmia_out/WikiMIA-25/}"

  IFS='/' read -r -a parts <<< "$rel"
  [[ ${#parts[@]} -eq 4 ]] || return 1

  SUB_DATASET="${parts[0]}"
  MODEL="${parts[1]}"
  NUM_SHOTS="${parts[2]}"
}

run_one_setting() {
  local cache_dir="$1"
  local sub_dataset="$2"
  local model="$3"
  local num_shots="$4"
  local tau="$5"
  local gamma="$6"
  local csv_path="$7"
  local scenario_slug="$8"
  local output_dir="$cache_dir"
  local tau_slug
  local gamma_slug
  local result_name
  local roc_path
  local log_file
  local status

  tau_slug="$(format_param_slug "$tau")"
  gamma_slug="$(format_param_slug "$gamma")"
  result_name="wpmia_${scenario_slug}_tau_${tau_slug}_gamma_${gamma_slug}"
  roc_path="${output_dir}/${result_name}_roc_tpr_at_5_fpr.png"
  log_file="${RUN_ROOT}/$(slugify "${scenario_slug}_tau_${tau_slug}_gamma_${gamma_slug}").log"

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
    --num_shots "$num_shots"
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

  log "START ${scenario_slug} tau=${tau} gamma=${gamma}"
  {
    printf 'COMMAND:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
  } > "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    append_csv_row "$csv_path" "dry_run" "$sub_dataset" "$model" "$num_shots" \
      "$tau" "$gamma" "$result_name" "$output_dir" "$roc_path" "$log_file"
    log "DONE  ${scenario_slug} tau=${tau} gamma=${gamma} status=dry_run"
    return 0
  fi

  status="ok"
  if ! "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    status="failed"
  fi

  append_csv_row "$csv_path" "$status" "$sub_dataset" "$model" "$num_shots" \
    "$tau" "$gamma" "$result_name" "$output_dir" "$roc_path" "$log_file"

  log "DONE  ${scenario_slug} tau=${tau} gamma=${gamma} status=${status}"
  if [[ "$status" != "ok" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
}

log "Run root: $RUN_ROOT"
log "CSV root: $CSV_ROOT"
log "WikiMIA-25 cache root: $CACHE_ROOT"
log "Tau values: $TAUS"
log "Gamma values: $GAMMAS"

if [[ ! -d "$CACHE_ROOT" ]]; then
  log "ERROR missing WikiMIA-25 cache root: $CACHE_ROOT"
  exit 2
fi

total_caches=0
while IFS= read -r records_path; do
  parse_wikimia25_cache_path "$records_path" || {
    log "SKIP unrecognized WikiMIA-25 cache path: $records_path"
    continue
  }

  cache_dir="$(dirname "$records_path")"
  csv_path="$(csv_for_model "$MODEL")"
  scenario_slug="$(slugify "wikimia25_${MODEL}_${SUB_DATASET}_${NUM_SHOTS}")"

  if ! cache_check_output="$(check_cache_complete "$cache_dir" 2>&1)"; then
    log "SKIP incomplete cache: $cache_dir"
    log "$cache_check_output"
    append_csv_row "$csv_path" "skipped_incomplete_cache" "$SUB_DATASET" \
      "$MODEL" "$NUM_SHOTS" "" "" "" "$cache_dir" "" "$SUMMARY_LOG"
    continue
  fi

  total_caches=$((total_caches + 1))
  log "CACHE $cache_dir"
  log "CSV   $csv_path"

  for tau in $TAUS; do
    for gamma in $GAMMAS; do
      run_one_setting "$cache_dir" "$SUB_DATASET" "$MODEL" "$NUM_SHOTS" \
        "$tau" "$gamma" "$csv_path" "$scenario_slug"
    done
  done
done < <(find "$CACHE_ROOT" -path '*/records.jsonl' -print | sort)

log "Finished. Swept $total_caches complete WikiMIA-25 caches."
