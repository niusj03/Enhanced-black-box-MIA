#!/usr/bin/env bash
set -euo pipefail

# Randomness ablation for WikiMIA on Pythia-6.9B.
#
# It studies two randomness sources separately:
#   1. prefix randomness: vary prefix_seed, keep sampling_seed fixed.
#   2. sampling randomness: vary sampling_seed, keep prefix_data/full_dataset fixed.
#
# Full run:
#   nohup bash scripts/run_wikimia_pythia_randomness_ablation.sh "0 1 2 3 4 5 6 7" \
#     > logs/ablation/wikimia_pythia69b_randomness.nohup.log 2>&1 &
#
# Small dry run:
#   DRY_RUN=1 LENGTHS="32" SEEDS="42 43" METHODS="wpmia" \
#   bash scripts/run_wikimia_pythia_randomness_ablation.sh "0"

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

MODEL="${MODEL:-EleutherAI/pythia-6.9b}"
DATA="${DATA:-swj0419/WikiMIA}"
LENGTHS="${LENGTHS:-32 64 128}"
SEEDS="${SEEDS:-42 43 44 45 46}"
METHODS="${METHODS:-simmia_star simmia wpmia}"
NUM_SHOTS="${NUM_SHOTS:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
PREFIX_RATIO="${PREFIX_RATIO:-0.0}"
TOP_K="${TOP_K:-20}"
WPMIA_TAU="${WPMIA_TAU:-0.2}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
FIXED_PREFIX_SEED="${FIXED_PREFIX_SEED:-42}"
FIXED_SAMPLING_SEED="${FIXED_SAMPLING_SEED:-42}"
BASELINE_ROOT="${BASELINE_ROOT:-simmia_out}"
ABLATION_ROOT="${ABLATION_ROOT:-ablation/randomness}"
CSV_DIR="${CSV_DIR:-logs/ablation}"
PREFIX_CSV="${PREFIX_CSV:-${CSV_DIR}/wikimia_pythia69b_prefix_randomness.csv}"
SAMPLING_CSV="${SAMPLING_CSV:-${CSV_DIR}/wikimia_pythia69b_sampling_randomness.csv}"
RUN_ROOT="${RUN_ROOT:-${CSV_DIR}/$(date +%Y%m%d_%H%M%S)_wikimia_pythia69b_randomness}"
SCORE_ROOT="${SCORE_ROOT:-${CSV_DIR}/wikimia_pythia69b_randomness_scores}"
SIMMIA_BIN="${SIMMIA_BIN:-simmia.benchmark}"
SAMPLING_BATCH_SIZE="${SAMPLING_BATCH_SIZE:-25}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
DRY_RUN="${DRY_RUN:-0}"
CACHE_ONLY="${CACHE_ONLY:-0}"
RESET_CSV="${RESET_CSV:-0}"
REUSE_BASELINE_CACHE="${REUSE_BASELINE_CACHE:-1}"
STD_DDOF="${STD_DDOF:-1}"

mkdir -p "$CSV_DIR" "$RUN_ROOT" "$SCORE_ROOT"
if [[ "$RESET_CSV" == "1" ]]; then
  rm -f "$PREFIX_CSV" "$SAMPLING_CSV"
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
    simmia_star) printf "SimMIA*" ;;
    simmia) printf "SimMIA" ;;
    wpmia) printf "WPMIA" ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

method_postprocess() {
  case "$1" in
    simmia_star) printf "process_relative_label_word_data" ;;
    simmia) printf "process_relative_word_data" ;;
    wpmia) printf "process_wpmia_word_data" ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
}

method_inference() {
  case "$1" in
    simmia_star) printf "relative_label_ratio" ;;
    simmia) printf "relative_semantic_ratio" ;;
    wpmia) printf "wpmia_score" ;;
    *) echo "Unknown method: $1" >&2; return 2 ;;
  esac
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
    raise SystemExit("cache is missing required fields: " + ", ".join(missing))
PY
}

copy_file_if_missing() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" && ! -f "$dst" ]]; then
    cp "$src" "$dst"
  fi
}

reuse_baseline_cache() {
  local length="$1"
  local target_dir="$2"
  local include_records="$3"
  local sub_dataset="WikiMIA_length${length}"
  local model_base="${MODEL##*/}"
  local data_base="${DATA##*/}"
  local baseline_dir="${BASELINE_ROOT}/${data_base}/${sub_dataset}/${model_base}/${NUM_SHOTS}"

  mkdir -p "$target_dir"
  copy_file_if_missing "${baseline_dir}/full_dataset.jsonl" "${target_dir}/full_dataset.jsonl"
  copy_file_if_missing "${baseline_dir}/prefix_data.json" "${target_dir}/prefix_data.json"

  if [[ "$include_records" == "1" ]]; then
    if check_cache_complete "$baseline_dir" >/dev/null 2>&1; then
      copy_file_if_missing "${baseline_dir}/records.jsonl" "${target_dir}/records.jsonl"
    fi
  fi
}

append_csv_row() {
  local csv_path="$1"
  local status="$2"
  local randomness="$3"
  local method="$4"
  local length="$5"
  local seed="$6"
  local prefix_seed="$7"
  local sampling_seed="$8"
  local result_name="$9"
  local output_dir="${10}"
  local roc_path="${11}"
  local score_path="${12}"
  local log_file="${13}"

  python - "$csv_path" "$status" "$randomness" "$method" "$(method_label "$method")" \
    "$length" "$seed" "$prefix_seed" "$sampling_seed" "$MODEL" "$DATA" \
    "$NUM_SHOTS" "$NUM_SAMPLES" "$PREFIX_RATIO" "$WPMIA_TAU" "$WPMIA_GAMMA" \
    "$result_name" "$output_dir" "$roc_path" "$score_path" "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    randomness,
    method,
    method_label,
    length,
    seed,
    prefix_seed,
    sampling_seed,
    model,
    data,
    num_shots,
    num_samples,
    prefix_ratio,
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

def metrics_from_score_dump(path):
    import numpy as np
    from sklearn.metrics import auc as auc_fn
    from sklearn.metrics import roc_curve

    labels = []
    scores = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float))
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    def low_at(threshold):
        idx = np.where(fpr < threshold)[0]
        return float(tpr[idx[-1]]) if len(idx) else 0.0
    return {
        "auc_pct": f"{float(auc_fn(fpr, tpr)) * 100:.6f}",
        "accuracy_pct": f"{float(acc) * 100:.6f}",
        "tpr1_fpr_pct": f"{low_at(0.01) * 100:.6f}",
        "tpr5_fpr_pct": f"{low_at(0.05) * 100:.6f}",
        "tpr10_fpr_pct": f"{low_at(0.10) * 100:.6f}",
    }

if status == "ok" and os.path.exists(score_path):
    try:
        metrics = metrics_from_score_dump(score_path)
    except Exception:
        pass

if not metrics["auc_pct"]:
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

fieldnames = [
    "row_type",
    "randomness",
    "method",
    "method_label",
    "length",
    "seed",
    "prefix_seed",
    "sampling_seed",
    "status",
    "run_count",
    "model",
    "data",
    "sub_dataset",
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
    "auc_mean_pct",
    "auc_std_pct",
    "accuracy_mean_pct",
    "accuracy_std_pct",
    "tpr1_fpr_mean_pct",
    "tpr1_fpr_std_pct",
    "tpr5_fpr_mean_pct",
    "tpr5_fpr_std_pct",
    "tpr10_fpr_mean_pct",
    "tpr10_fpr_std_pct",
    "result_name",
    "output_dir",
    "roc_path",
    "score_path",
    "log_file",
]

row = {
    "row_type": "run",
    "randomness": randomness,
    "method": method,
    "method_label": method_label,
    "length": length,
    "seed": seed,
    "prefix_seed": prefix_seed,
    "sampling_seed": sampling_seed,
    "status": status,
    "run_count": "",
    "model": model,
    "data": data,
    "sub_dataset": f"WikiMIA_length{length}",
    "num_shots": num_shots,
    "num_samples": num_samples,
    "prefix_ratio": prefix_ratio,
    "wpmia_tau": tau if method == "wpmia" else "",
    "wpmia_gamma": gamma if method == "wpmia" else "",
    "result_name": result_name,
    "output_dir": output_dir,
    "roc_path": roc_path,
    "score_path": score_path,
    "log_file": log_file,
    **metrics,
}

write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, "a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

write_summary_rows() {
  local csv_path="$1"
  python - "$csv_path" "$STD_DDOF" <<'PY'
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

csv_path, std_ddof = sys.argv[1:]
std_ddof = int(std_ddof)
if not os.path.exists(csv_path):
    raise SystemExit(0)

with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if row.get("row_type") != "summary"]

metrics = [
    ("auc_pct", "auc_mean_pct", "auc_std_pct"),
    ("accuracy_pct", "accuracy_mean_pct", "accuracy_std_pct"),
    ("tpr1_fpr_pct", "tpr1_fpr_mean_pct", "tpr1_fpr_std_pct"),
    ("tpr5_fpr_pct", "tpr5_fpr_mean_pct", "tpr5_fpr_std_pct"),
    ("tpr10_fpr_pct", "tpr10_fpr_mean_pct", "tpr10_fpr_std_pct"),
]

groups = defaultdict(list)
for row in rows:
    if row.get("row_type") != "run" or row.get("status") != "ok":
        continue
    if not row.get("auc_pct"):
        continue
    key = (row["randomness"], row["method"], row["method_label"], row["length"])
    groups[key].append(row)

summary_rows = []
for (randomness, method, method_label, length), group in sorted(
    groups.items(), key=lambda item: (item[0][0], int(item[0][3]), item[0][1])
):
    base = group[0]
    summary = {field: "" for field in fieldnames}
    summary.update(
        {
            "row_type": "summary",
            "randomness": randomness,
            "method": method,
            "method_label": method_label,
            "length": length,
            "status": "ok",
            "run_count": str(len(group)),
            "model": base.get("model", ""),
            "data": base.get("data", ""),
            "sub_dataset": base.get("sub_dataset", ""),
            "num_shots": base.get("num_shots", ""),
            "num_samples": base.get("num_samples", ""),
            "prefix_ratio": base.get("prefix_ratio", ""),
            "wpmia_tau": base.get("wpmia_tau", ""),
            "wpmia_gamma": base.get("wpmia_gamma", ""),
        }
    )
    for raw_col, mean_col, std_col in metrics:
        values = [float(row[raw_col]) for row in group if row.get(raw_col)]
        if not values:
            continue
        mean = statistics.fmean(values)
        if len(values) <= 1:
            std = 0.0
        elif std_ddof == 0:
            std = statistics.pstdev(values)
        else:
            std = statistics.stdev(values)
        summary[mean_col] = f"{mean:.6f}"
        summary[std_col] = f"{std:.6f}"
    summary_rows.append(summary)

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    writer.writerows(summary_rows)
PY
}

run_one() {
  local randomness="$1"
  local length="$2"
  local seed="$3"
  local method="$4"
  local output_root="$5"
  local prefix_seed="$6"
  local sampling_seed="$7"
  local csv_path="$8"

  local sub_dataset="WikiMIA_length${length}"
  local model_base="${MODEL##*/}"
  local data_base="${DATA##*/}"
  local output_dir="${output_root}/${data_base}/${sub_dataset}/${model_base}/${NUM_SHOTS}"
  local method_slug result_name roc_path score_path log_file status
  local -a method_extra=()

  method_slug="$(slugify "$method")"
  result_name="${method_slug}_randomness_${randomness}_seed_${seed}_len${length}"
  if [[ "$method" == "wpmia" ]]; then
    result_name="${result_name}_tau_$(format_param_slug "$WPMIA_TAU")_gamma_$(format_param_slug "$WPMIA_GAMMA")"
    method_extra+=(--wpmia_tau "$WPMIA_TAU" --wpmia_gamma "$WPMIA_GAMMA")
  fi
  if [[ "$method" == "simmia_star" ]]; then
    method_extra+=(--smoothing)
  fi

  roc_path="${output_dir}/${result_name}_roc_tpr_at_5_fpr.png"
  score_path="${SCORE_ROOT}/${randomness}_len${length}_seed${seed}_${method_slug}.csv"
  log_file="${RUN_ROOT}/${randomness}_len${length}_seed${seed}_${method_slug}.log"

  if [[ "$CACHE_ONLY" == "1" ]]; then
    local cache_message
    if ! cache_message="$(check_cache_complete "$output_dir" 2>&1)"; then
      status="missing_cache"
      {
        echo "[SKIP] ${randomness} len=${length} seed=${seed} method=${method}"
        echo "$cache_message"
      } | tee "$log_file"
      append_csv_row "$csv_path" "$status" "$randomness" "$method" "$length" "$seed" \
        "$prefix_seed" "$sampling_seed" "$result_name" "$output_dir" "$roc_path" \
        "$score_path" "$log_file"
      return 0
    fi
  fi

  read -r -a simmia_cmd <<< "$SIMMIA_BIN"
  cmd=(
    "${simmia_cmd[@]}"
    --model_name_or_path "$MODEL"
    --sampling relative_word_by_word
    --postprocess "$(method_postprocess "$method")"
    --inference "$(method_inference "$method")"
    --output_dir "$output_root"
    --result_name "$result_name"
    --num_samples "$NUM_SAMPLES"
    --data "$DATA"
    --sub_dataset "$sub_dataset"
    --num_shots "$NUM_SHOTS"
    --prefix_ratio "$PREFIX_RATIO"
    --top_k "$TOP_K"
    --seed "$seed"
    --prefix_seed "$prefix_seed"
    --sampling_seed "$sampling_seed"
    --score_dump_path "$score_path"
  )
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

  echo "[START] randomness=${randomness} length=${length} seed=${seed} method=${method}"
  {
    echo "COMMAND: ${cmd[*]}"
    echo
  } > "$log_file"

  status="ok"
  if [[ "$DRY_RUN" == "1" ]]; then
    status="dry_run"
    echo "[DRY_RUN] ${cmd[*]}" | tee -a "$log_file"
  elif ! "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    status="failed"
  fi

  append_csv_row "$csv_path" "$status" "$randomness" "$method" "$length" "$seed" \
    "$prefix_seed" "$sampling_seed" "$result_name" "$output_dir" "$roc_path" \
    "$score_path" "$log_file"

  echo "[DONE] randomness=${randomness} length=${length} seed=${seed} method=${method} status=${status}"
  if [[ "$status" == "failed" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "Stopping after failed run. Set CONTINUE_ON_ERROR=1 to keep going." >&2
    exit 1
  fi
}

prepare_dimension_cache() {
  local randomness="$1"
  local length="$2"
  local seed="$3"
  local output_root="$4"
  local sub_dataset="WikiMIA_length${length}"
  local model_base="${MODEL##*/}"
  local data_base="${DATA##*/}"
  local cache_dir="${output_root}/${data_base}/${sub_dataset}/${model_base}/${NUM_SHOTS}"

  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  if [[ "$REUSE_BASELINE_CACHE" != "1" ]]; then
    return 0
  fi

  if [[ "$randomness" == "sampling" ]]; then
    local include_records="0"
    [[ "$seed" == "$FIXED_SAMPLING_SEED" ]] && include_records="1"
    reuse_baseline_cache "$length" "$cache_dir" "$include_records"
  elif [[ "$randomness" == "prefix" && "$seed" == "$FIXED_PREFIX_SEED" ]]; then
    reuse_baseline_cache "$length" "$cache_dir" "1"
  fi
}

run_dimension() {
  local randomness="$1"
  local csv_path="$2"
  local length seed method output_root prefix_seed sampling_seed

  for length in $LENGTHS; do
    for seed in $SEEDS; do
      if [[ "$randomness" == "prefix" ]]; then
        output_root="${ABLATION_ROOT}/prefix_seed_${seed}"
        prefix_seed="$seed"
        sampling_seed="$FIXED_SAMPLING_SEED"
      else
        output_root="${ABLATION_ROOT}/sampling_seed_${seed}"
        prefix_seed="$FIXED_PREFIX_SEED"
        sampling_seed="$seed"
      fi

      prepare_dimension_cache "$randomness" "$length" "$seed" "$output_root"

      for method in $METHODS; do
        run_one "$randomness" "$length" "$seed" "$method" "$output_root" \
          "$prefix_seed" "$sampling_seed" "$csv_path"
      done
    done
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    write_summary_rows "$csv_path"
  fi
}

echo "Run root: $RUN_ROOT"
echo "Score root: $SCORE_ROOT"
echo "Prefix randomness CSV: $PREFIX_CSV"
echo "Sampling randomness CSV: $SAMPLING_CSV"
echo "Model: $MODEL"
echo "Lengths: $LENGTHS"
echo "Seeds: $SEEDS"
echo "Methods: $METHODS"
echo "Fixed prefix seed: $FIXED_PREFIX_SEED"
echo "Fixed sampling seed: $FIXED_SAMPLING_SEED"
echo "WPMIA tau/gamma: $WPMIA_TAU / $WPMIA_GAMMA"

run_dimension "prefix" "$PREFIX_CSV"
run_dimension "sampling" "$SAMPLING_CSV"

echo "Finished randomness ablation."
echo "CSV files:"
echo "  $PREFIX_CSV"
echo "  $SAMPLING_CSV"
