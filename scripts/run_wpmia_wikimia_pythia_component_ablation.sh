#!/usr/bin/env bash
set -euo pipefail

# Component-wise WPMIA ablation on WikiMIA / Pythia-6.9B.
#
# Usage:
#   nohup bash scripts/run_wpmia_wikimia_pythia_component_ablation.sh "0 1 2 3 4 5 6 7" \
#     > logs/ablation/wikimia_pythia69b_component_wise.nohup.log 2>&1 &
#
# Useful overrides:
#   LENGTHS="32 64 128"
#   COMPONENT_METHODS="ll seq_nm_ratio seq_contrast word_nm_ratio word_contrast"
#   WPMIA_TAU=0.2
#   WPMIA_GAMMA=1.0
#   CACHE_ONLY=1
#   DRY_RUN=1

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

MODEL="${MODEL:-EleutherAI/pythia-6.9b}"
DATA="${DATA:-swj0419/WikiMIA}"
LENGTHS="${LENGTHS:-32 64 128}"
NUM_SHOTS="${NUM_SHOTS:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-simmia_out}"
WPMIA_TAU="${WPMIA_TAU:-0.2}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
COMPONENT_METHODS="${COMPONENT_METHODS:-ll seq_nm_ratio seq_contrast word_nm_ratio word_contrast}"
CACHE_ONLY="${CACHE_ONLY:-1}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RESET_CSV="${RESET_CSV:-0}"
SAMPLING_BATCH_SIZE="${SAMPLING_BATCH_SIZE:-25}"
SIMMIA_BIN="${SIMMIA_BIN:-simmia.benchmark}"

RUN_ROOT="${RUN_ROOT:-logs/ablation/$(date +%Y%m%d_%H%M%S)_wikimia_pythia69b_component_wise}"
CSV_PATH="${CSV_PATH:-logs/ablation/wikimia_pythia69b_component_wise.csv}"
SCORE_ROOT="${SCORE_ROOT:-logs/ablation/wikimia_pythia69b_component_scores}"

mkdir -p "$RUN_ROOT" "$(dirname "$CSV_PATH")" "$SCORE_ROOT"
if [[ "$RESET_CSV" == "1" ]]; then
  rm -f "$CSV_PATH"
fi

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

method_label() {
  case "$1" in
    ll) printf "Log-likelihood" ;;
    seq_nm_ratio) printf "Recall" ;;
    seq_contrast) printf "Con-Recall" ;;
    word_nm_ratio) printf "Word Recall" ;;
    word_contrast) printf "Word Con-Recall" ;;
    *) echo "Unknown component method: $1" >&2; return 2 ;;
  esac
}

method_inference() {
  case "$1" in
    ll) printf "wpmia_ll_score" ;;
    seq_nm_ratio) printf "wpmia_nonmember_ratio_score" ;;
    seq_contrast) printf "wpmia_score" ;;
    word_nm_ratio) printf "wpmia_word_nonmember_ratio_score" ;;
    word_contrast) printf "wpmia_word_score" ;;
    *) echo "Unknown component method: $1" >&2; return 2 ;;
  esac
}

append_csv_row() {
  local status="$1"
  local method="$2"
  local label="$3"
  local length="$4"
  local result_name="$5"
  local output_dir="$6"
  local roc_path="$7"
  local score_path="$8"
  local log_file="$9"

  python - "$CSV_PATH" "$status" "$method" "$label" "$length" "$MODEL" "$DATA" \
    "$NUM_SHOTS" "$WPMIA_TAU" "$WPMIA_GAMMA" "$result_name" "$output_dir" \
    "$roc_path" "$score_path" "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    method,
    method_label,
    length,
    model,
    data,
    num_shots,
    tau,
    gamma,
    result_name,
    output_dir,
    roc_path,
    score_path,
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
    "method": method,
    "method_label": method_label,
    "length": length,
    "model": model,
    "data": data,
    "sub_dataset": f"WikiMIA_length{length}",
    "num_shots": num_shots,
    "wpmia_tau": tau,
    "wpmia_gamma": gamma,
    "result_name": result_name,
    "output_dir": output_dir,
    "roc_path": roc_path,
    "score_path": score_path,
    "log_file": log_file,
    **metrics,
}
fieldnames = [
    "status",
    "method",
    "method_label",
    "length",
    "model",
    "data",
    "sub_dataset",
    "num_shots",
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
    "score_path",
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
  python - "$cache_dir" <<'PY'
import json
import os
import sys

cache_dir = sys.argv[1]
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
required = [
    "label_results",
    "sample_results",
    "nonmember_prefix_sample_results",
    "member_prefix_sample_results",
]
missing = [field for field in required if field not in first]
if missing:
    raise SystemExit("cache is missing required WPMIA fields: " + ", ".join(missing))
PY
}

run_one() {
  local length="$1"
  local method="$2"
  local sub_dataset="WikiMIA_length${length}"
  local model_base="${MODEL##*/}"
  local data_base="${DATA##*/}"
  local output_path="${OUTPUT_DIR}/${data_base}/${sub_dataset}/${model_base}/${NUM_SHOTS}"
  local label inference tau_slug gamma_slug result_name roc_path score_path log_file status

  label="$(method_label "$method")"
  inference="$(method_inference "$method")"
  tau_slug="$(format_param_slug "$WPMIA_TAU")"
  gamma_slug="$(format_param_slug "$WPMIA_GAMMA")"
  result_name="wpmia_component_len${length}_${method}_tau_${tau_slug}_gamma_${gamma_slug}"
  roc_path="${output_path}/${result_name}_roc_tpr_at_5_fpr.png"
  score_path="${SCORE_ROOT}/wikimia_len${length}_${method}.csv"
  log_file="${RUN_ROOT}/$(slugify "wikimia_len${length}_${method}").log"
  status="ok"

  if [[ "$CACHE_ONLY" == "1" ]]; then
    local cache_message
    if ! cache_message="$(check_cache_complete "$output_path" 2>&1)"; then
      status="missing_cache"
      {
        echo "[SKIP] length=${length} method=${method}"
        echo "$cache_message"
      } | tee "$log_file"
      append_csv_row "$status" "$method" "$label" "$length" "$result_name" \
        "$output_path" "$roc_path" "$score_path" "$log_file"
      return 0
    fi
  fi

  read -r -a simmia_cmd <<< "$SIMMIA_BIN"
  cmd=(
    "${simmia_cmd[@]}"
    --model_name_or_path "$MODEL"
    --sampling relative_word_by_word
    --postprocess process_wpmia_word_data
    --inference "$inference"
    --output_dir "$OUTPUT_DIR"
    --result_name "$result_name"
    --score_dump_path "$score_path"
    --num_samples "$NUM_SAMPLES"
    --data "$DATA"
    --sub_dataset "$sub_dataset"
    --num_shots "$NUM_SHOTS"
    --prefix_ratio 0.0
    --top_k 20
    --wpmia_tau "$WPMIA_TAU"
    --wpmia_gamma "$WPMIA_GAMMA"
  )
  if [[ -n "$SAMPLING_BATCH_SIZE" ]]; then
    cmd+=(--params "sampling_batch_size:${SAMPLING_BATCH_SIZE}")
  fi
  if [[ -n "${GPU_IDS:-}" ]]; then
    local gpu_ids_norm
    gpu_ids_norm="${GPU_IDS//,/ }"
    read -r -a gpu_id_arr <<< "$gpu_ids_norm"
    cmd+=(--gpu_ids "${gpu_id_arr[@]}")
  fi
  [[ -n "${CONCURRENCY:-}" ]] && cmd+=(--concurrency "$CONCURRENCY")
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[START] length=${length} method=${method} label=${label}"
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

  append_csv_row "$status" "$method" "$label" "$length" "$result_name" \
    "$output_path" "$roc_path" "$score_path" "$log_file"

  echo "[DONE] length=${length} method=${method} status=${status}"
  if [[ "$status" == "failed" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "Stopping after failed run. Set CONTINUE_ON_ERROR=1 to continue." >&2
    exit 1
  fi
}

echo "Run root: $RUN_ROOT"
echo "CSV: $CSV_PATH"
echo "Score root: $SCORE_ROOT"
echo "Model: $MODEL"
echo "Lengths: $LENGTHS"
echo "Methods: $COMPONENT_METHODS"
echo "WPMIA tau/gamma: $WPMIA_TAU / $WPMIA_GAMMA"
echo "Cache only: $CACHE_ONLY"

for length in $LENGTHS; do
  for method in $COMPONENT_METHODS; do
    run_one "$length" "$method"
  done
done

echo "Finished. Summary CSV written to $CSV_PATH"
