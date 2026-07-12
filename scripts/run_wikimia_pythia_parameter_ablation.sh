#!/usr/bin/env bash
set -euo pipefail

# Parameter ablations on WikiMIA length 32 for Pythia-6.9B.
# Compares SimMIA*, SimMIA, and WPMIA across prefix ratio, cached sample count,
# and number of shots.
#
# Usage:
#   nohup bash scripts/run_wikimia_pythia_parameter_ablation.sh "0 1 2 3 4 5 6 7" \
#     > logs/ablation/wikimia32_pythia69b_parameter_ablation.nohup.log 2>&1 &
#
# Useful overrides:
#   MODEL="EleutherAI/pythia-6.9b"
#   ABLATION_ELEMENTS="prefix_ratio num_samples num_shots"
#   ABLATION_METHODS="simmia_star simmia wpmia"
#   PREFIX_RATIOS="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
#   SAMPLE_COUNTS="10 20 30 40 50 60 70 80 90 100"
#   SHOT_COUNTS="1 2 3 4 5 6 7 8 9 10"
#   CACHE_ONLY=1
#   DRY_RUN=1

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

MODEL="${MODEL:-${WPMIA_MODEL:-EleutherAI/pythia-6.9b}}"
DATA="${DATA:-${WPMIA_DATA:-swj0419/WikiMIA}}"
NUM_SHOTS="${NUM_SHOTS:-7}"
CACHE_NUM_SAMPLES="${CACHE_NUM_SAMPLES:-100}"
SAMPLING_BATCH_SIZE="${SAMPLING_BATCH_SIZE:-25}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SIMMIA_BIN="${SIMMIA_BIN:-simmia.benchmark}"

slugify() {
  local s="$1"
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

model_label() {
  local base="$1"
  case "$base" in
    pythia-6.9b) printf "pythia69b" ;;
    opt-6.7b) printf "opt67b" ;;
    llama-13b) printf "llama13b" ;;
    gpt-neox-20b) printf "gptneox20b" ;;
    *) slugify "${base,,}" | sed 's/p//g' ;;
  esac
}

append_ablation_csv_row() {
  local csv_path="$1"
  local status="$2"
  local ablation="$3"
  local ablation_value="$4"
  local method="$5"
  local length="$6"
  local num_shots="$7"
  local num_samples="$8"
  local prefix_ratio="$9"
  local tau="${10}"
  local gamma="${11}"
  local result_name="${12}"
  local output_dir="${13}"
  local roc_path="${14}"
  local log_file="${15}"

  python - "$csv_path" "$status" "$ablation" "$ablation_value" "$method" \
    "$MODEL" "$DATA" "$length" "$num_shots" "$num_samples" "$prefix_ratio" \
    "$tau" "$gamma" "$result_name" "$output_dir" "$roc_path" "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    ablation,
    ablation_value,
    method,
    model,
    data,
    length,
    num_shots,
    num_samples,
    prefix_ratio,
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
    "ablation": ablation,
    "ablation_value": ablation_value,
    "method": method,
    "model": model,
    "data": data,
    "sub_dataset": f"WikiMIA_length{length}",
    "length": length,
    "num_shots": num_shots,
    "num_samples": num_samples,
    "prefix_ratio": prefix_ratio,
    "wpmia_tau": tau,
    "wpmia_gamma": gamma,
    "result_name": result_name,
    "output_dir": output_dir,
    "roc_path": roc_path,
    "log_file": log_file,
    **metrics,
}

fieldnames = [
    "status",
    "ablation",
    "ablation_value",
    "method",
    "model",
    "data",
    "sub_dataset",
    "length",
    "num_shots",
    "num_samples",
    "prefix_ratio",
    "wpmia_tau",
    "wpmia_gamma",
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

check_records_cache() {
  local cache_dir="$1"
  local method="$2"
  python - "$cache_dir" "$method" <<'PY'
import json
import os
import sys

cache_dir, method = sys.argv[1:]
records_path = os.path.join(cache_dir, "records.jsonl")
full_path = os.path.join(cache_dir, "full_dataset.jsonl")

if not os.path.exists(records_path):
    raise SystemExit(f"missing records.jsonl: {records_path}")
if os.path.getsize(records_path) == 0:
    raise SystemExit(f"empty records.jsonl: {records_path}")

with open(records_path, "r", encoding="utf-8") as f:
    record_lines = [line for line in f if line.strip()]

if os.path.exists(full_path):
    with open(full_path, "r", encoding="utf-8") as f:
        full_lines = [line for line in f if line.strip()]
    if len(record_lines) < len(full_lines):
        raise SystemExit(
            f"incomplete cache: {len(record_lines)} records for {len(full_lines)} rows in {cache_dir}"
        )

first = json.loads(record_lines[0])
required = ["label_results", "sample_results", "nonmember_prefix_sample_results"]
if method == "wpmia":
    required.append("member_prefix_sample_results")
missing = [field for field in required if field not in first]
if missing:
    raise SystemExit(f"cache is missing required fields for {method}: {', '.join(missing)}")
PY
}

csv_for_ablation() {
  local ablation="$1"
  local method="$2"
  printf "%s/%s_%s_%s.csv" "$ABLATION_CSV_DIR" "$ABLATION_CSV_PREFIX" "$ablation" "$method"
}

method_result_prefix() {
  local method="$1"
  case "$method" in
    simmia_star) printf "simmia_star" ;;
    simmia) printf "simmia" ;;
    wpmia) printf "wpmia" ;;
    *) echo "Unknown ablation method: $method" >&2; return 2 ;;
  esac
}

run_ablation_eval() {
  local ablation="$1"
  local ablation_value="$2"
  local method="$3"
  local output_root="$4"
  local num_shots="$5"
  local prefix_ratio="$6"
  local sample_count_limit="$7"
  local num_samples_for_csv="$8"

  local method_prefix value_slug result_name csv_path cache_dir output_dir roc_path log_file
  local status postprocess inference tau gamma
  local -a method_extra=()

  method_prefix="$(method_result_prefix "$method")"
  value_slug="$(format_param_slug "$ablation_value")"
  tau=""
  gamma=""

  case "$method" in
    simmia_star)
      postprocess="process_relative_label_word_data"
      inference="relative_label_ratio"
      method_extra+=(--smoothing)
      ;;
    simmia)
      postprocess="process_relative_word_data"
      inference="relative_semantic_ratio"
      ;;
    wpmia)
      postprocess="process_wpmia_word_data"
      inference="wpmia_score"
      tau="$ABLATION_WPMIA_TAU"
      gamma="$ABLATION_WPMIA_GAMMA"
      method_extra+=(--wpmia_tau "$tau" --wpmia_gamma "$gamma")
      ;;
  esac

  result_name="${method_prefix}_ablation_${ablation}_${value_slug}"
  if [[ "$method" == "wpmia" ]]; then
    result_name="${result_name}_tau_$(format_param_slug "$ABLATION_WPMIA_TAU")_gamma_$(format_param_slug "$ABLATION_WPMIA_GAMMA")"
  fi

  cache_dir="${output_root}/${data_base}/${ABLATION_SUB_DATASET}/${model_base}/${num_shots}"
  output_dir="$cache_dir"
  roc_path="${output_dir}/${result_name}_roc_tpr_at_5_fpr.png"
  log_file="${ABLATION_RUN_ROOT}/$(slugify "${ablation}_${ablation_value}_${method}").log"
  csv_path="$(csv_for_ablation "$ablation" "$method")"

  status="ok"
  if [[ "$CACHE_ONLY" == "1" ]]; then
    local cache_message
    if ! cache_message="$(check_records_cache "$cache_dir" "$method" 2>&1)"; then
      status="missing_cache"
      {
        echo "[SKIP] ${method} ${ablation}=${ablation_value}"
        echo "$cache_message"
      } | tee "$log_file"
      append_ablation_csv_row "$csv_path" "$status" "$ablation" "$ablation_value" "$method" \
        "$ABLATION_LENGTH" "$num_shots" "$num_samples_for_csv" "$prefix_ratio" "$tau" "$gamma" \
        "$result_name" "$output_dir" "$roc_path" "$log_file"
      return 0
    fi
  fi

  read -r -a simmia_cmd <<< "$SIMMIA_BIN"
  cmd=(
    "${simmia_cmd[@]}"
    --model_name_or_path "$MODEL"
    --sampling relative_word_by_word
    --postprocess "$postprocess"
    --inference "$inference"
    --output_dir "$output_root"
    --result_name "$result_name"
    --num_samples "$CACHE_NUM_SAMPLES"
    --data "$DATA"
    --sub_dataset "$ABLATION_SUB_DATASET"
    --num_shots "$num_shots"
    --prefix_ratio "$prefix_ratio"
    --top_k 20
  )
  if [[ -n "$sample_count_limit" ]]; then
    cmd+=(--sample_count_limit "$sample_count_limit")
  fi
  if [[ -n "$SAMPLING_BATCH_SIZE" ]]; then
    cmd+=(--params "sampling_batch_size:${SAMPLING_BATCH_SIZE}")
  fi
  cmd+=("${method_extra[@]}")

  if [[ -n "${GPU_IDS:-}" ]]; then
    local gpu_ids_norm
    gpu_ids_norm="${GPU_IDS//,/ }"
    read -r -a gpu_id_arr <<< "$gpu_ids_norm"
    cmd+=(--gpu_ids "${gpu_id_arr[@]}")
  fi
  [[ -n "${CONCURRENCY:-}" ]] && cmd+=(--concurrency "$CONCURRENCY")
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[START] ${method} ${ablation}=${ablation_value}"
  {
    echo "COMMAND: ${cmd[*]}"
    echo
  } > "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    status="dry_run"
    echo "[DRY_RUN] ${cmd[*]}" | tee -a "$log_file"
  elif ! "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    status="failed"
  fi

  append_ablation_csv_row "$csv_path" "$status" "$ablation" "$ablation_value" "$method" \
    "$ABLATION_LENGTH" "$num_shots" "$num_samples_for_csv" "$prefix_ratio" "$tau" "$gamma" \
    "$result_name" "$output_dir" "$roc_path" "$log_file"

  echo "[DONE] ${method} ${ablation}=${ablation_value} status=${status}"
  if [[ "$status" == "failed" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "Stopping after failed ablation run. Set CONTINUE_ON_ERROR=1 to keep sweeping." >&2
    exit 1
  fi
}

run_ablation() {
  ABLATION_LENGTH="${ABLATION_LENGTH:-32}"
  ABLATION_SUB_DATASET="${ABLATION_SUB_DATASET:-WikiMIA_length${ABLATION_LENGTH}}"
  ABLATION_ELEMENTS="${ABLATION_ELEMENTS:-prefix_ratio num_samples num_shots}"
  ABLATION_METHODS="${ABLATION_METHODS:-simmia_star simmia wpmia}"
  PREFIX_RATIOS="${PREFIX_RATIOS:-0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9}"
  SAMPLE_COUNTS="${SAMPLE_COUNTS:-10 20 30 40 50 60 70 80 90 100}"
  SHOT_COUNTS="${SHOT_COUNTS:-1 2 3 4 5 6 7 8 9 10}"
  MAIN_OUTPUT_ROOT="${MAIN_OUTPUT_ROOT:-simmia_out}"
  SHOTS_OUTPUT_ROOT="${SHOTS_OUTPUT_ROOT:-ablation}"
  CACHE_ONLY="${CACHE_ONLY:-1}"
  DRY_RUN="${DRY_RUN:-0}"
  RESET_CSV="${RESET_CSV:-0}"
  ABLATION_WPMIA_TAU="${ABLATION_WPMIA_TAU:-0.2}"
  ABLATION_WPMIA_GAMMA="${ABLATION_WPMIA_GAMMA:-1.0}"
  ABLATION_CSV_DIR="${ABLATION_CSV_DIR:-logs/ablation}"
  ABLATION_RUN_ROOT="${ABLATION_RUN_ROOT:-logs/ablation/$(date +%Y%m%d_%H%M%S)_wikimia32_pythia69b_ablation}"
  ABLATION_CSV_PREFIX="${ABLATION_CSV_PREFIX:-wikimia${ABLATION_LENGTH}_$(model_label "$model_base")}"

  mkdir -p "$ABLATION_RUN_ROOT" "$ABLATION_CSV_DIR"

  if [[ "$RESET_CSV" == "1" ]]; then
    local element method
    for element in $ABLATION_ELEMENTS; do
      for method in $ABLATION_METHODS; do
        rm -f "$(csv_for_ablation "$element" "$method")"
      done
    done
  fi

  echo "Ablation run root: $ABLATION_RUN_ROOT"
  echo "CSV dir: $ABLATION_CSV_DIR"
  echo "CSV prefix: $ABLATION_CSV_PREFIX"
  echo "Model: $MODEL"
  echo "Dataset: $DATA / $ABLATION_SUB_DATASET"
  echo "Methods: $ABLATION_METHODS"
  echo "Elements: $ABLATION_ELEMENTS"
  echo "WPMIA tau/gamma: $ABLATION_WPMIA_TAU / $ABLATION_WPMIA_GAMMA"
  echo "Cache only: $CACHE_ONLY"

  local element value method
  for element in $ABLATION_ELEMENTS; do
    case "$element" in
      prefix_ratio)
        for value in $PREFIX_RATIOS; do
          for method in $ABLATION_METHODS; do
            run_ablation_eval "prefix_ratio" "$value" "$method" "$MAIN_OUTPUT_ROOT" \
              "$NUM_SHOTS" "$value" "" "$CACHE_NUM_SAMPLES"
          done
        done
        ;;
      num_samples)
        for value in $SAMPLE_COUNTS; do
          for method in $ABLATION_METHODS; do
            run_ablation_eval "num_samples" "$value" "$method" "$MAIN_OUTPUT_ROOT" \
              "$NUM_SHOTS" "0.0" "$value" "$value"
          done
        done
        ;;
      num_shots)
        for value in $SHOT_COUNTS; do
          for method in $ABLATION_METHODS; do
            run_ablation_eval "num_shots" "$value" "$method" "$SHOTS_OUTPUT_ROOT" \
              "$value" "0.0" "" "$CACHE_NUM_SAMPLES"
          done
        done
        ;;
      *)
        echo "Unknown ablation element: $element" >&2
        exit 2
        ;;
    esac
  done

  echo "Finished ablation. CSV files are under $ABLATION_CSV_DIR"
}

model_base="${MODEL##*/}"
data_base="${DATA##*/}"
run_ablation
