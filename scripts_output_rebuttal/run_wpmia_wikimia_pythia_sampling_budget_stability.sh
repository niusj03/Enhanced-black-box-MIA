#!/usr/bin/env bash
set -euo pipefail

# Rebuttal experiment: WPMIA sampling-budget stability across five sampling seeds.
#
# Reuses the completed caches under:
#   ablation/randomness/sampling_seed_<seed>/WikiMIA/WikiMIA_length<length>/pythia-6.9b/7
#
# It never performs target-LLM sampling. For each cached seed and length, it
# evaluates sample_count_limit=10,20,...,100 with WPMIA tau=0.2, gamma=1.0.
#
# Full run:
#   RESET_CSV=1 nohup bash \
#     scripts_output_rebuttal/run_wpmia_wikimia_pythia_sampling_budget_stability.sh \
#     "0 1 2 3 4 5 6 7" \
#     > scripts_output_rebuttal/wikimia_pythia69b_sampling_budget.nohup.log 2>&1 &
#
# Small dry run:
#   DRY_RUN=1 LENGTHS="32" SEEDS="42 43" SAMPLE_COUNTS="10 20" \
#   bash scripts_output_rebuttal/run_wpmia_wikimia_pythia_sampling_budget_stability.sh "0"

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

MODEL="${MODEL:-EleutherAI/pythia-6.9b}"
DATA="${DATA:-swj0419/WikiMIA}"
LENGTHS="${LENGTHS:-32 64 128}"
SEEDS="${SEEDS:-42 43 44 45 46}"
SAMPLE_COUNTS="${SAMPLE_COUNTS:-10 20 30 40 50 60 70 80 90 100}"
NUM_SHOTS="${NUM_SHOTS:-7}"
CACHE_NUM_SAMPLES="${CACHE_NUM_SAMPLES:-100}"
PREFIX_RATIO="${PREFIX_RATIO:-0.0}"
TOP_K="${TOP_K:-20}"
FIXED_PREFIX_SEED="${FIXED_PREFIX_SEED:-42}"
WPMIA_TAU="${WPMIA_TAU:-0.2}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
CACHE_ROOT="${CACHE_ROOT:-ablation/randomness}"
OUTPUT_ROOT="${OUTPUT_ROOT:-scripts_output_rebuttal}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/logs/$(date +%Y%m%d_%H%M%S)_sampling_budget_stability}"
SCORE_ROOT="${SCORE_ROOT:-${OUTPUT_ROOT}/scores/wikimia_pythia69b_sampling_budget}"
ROC_ROOT="${ROC_ROOT:-${OUTPUT_ROOT}/roc/wikimia_pythia69b_sampling_budget}"
CSV_PATH="${CSV_PATH:-${OUTPUT_ROOT}/wikimia_pythia69b_wpmia_sampling_budget_stability.csv}"
SUMMARY_CSV_PATH="${SUMMARY_CSV_PATH:-${OUTPUT_ROOT}/wikimia_pythia69b_wpmia_sampling_budget_summary.csv}"
SIMMIA_BIN="${SIMMIA_BIN:-simmia.benchmark}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESET_CSV="${RESET_CSV:-0}"
DRY_RUN="${DRY_RUN:-0}"
STD_DDOF="${STD_DDOF:-1}"

mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$SCORE_ROOT" "$ROC_ROOT"
if [[ "$RESET_CSV" == "1" ]]; then
  rm -f "$CSV_PATH" "$SUMMARY_CSV_PATH"
fi

format_param_slug() {
  local s="$1"
  s="${s//./p}"
  s="${s//-/_neg_}"
  printf '%s' "$s"
}

cache_dir_for() {
  local length="$1"
  local seed="$2"
  local model_base="${MODEL##*/}"
  local data_base="${DATA##*/}"
  printf "%s/sampling_seed_%s/%s/WikiMIA_length%s/%s/%s" \
    "$CACHE_ROOT" "$seed" "$data_base" "$length" "$model_base" "$NUM_SHOTS"
}

check_cache_complete() {
  local cache_dir="$1"
  python - "$cache_dir" "$CACHE_NUM_SAMPLES" <<'PY'
import json
import os
import sys

cache_dir, expected_samples = sys.argv[1:]
expected_samples = int(expected_samples)
records_path = os.path.join(cache_dir, "records.jsonl")
full_path = os.path.join(cache_dir, "full_dataset.jsonl")
prefix_path = os.path.join(cache_dir, "prefix_data.json")

for path in (records_path, full_path, prefix_path):
    if not os.path.exists(path):
        raise SystemExit(f"missing cache file: {path}")

with open(records_path, "r", encoding="utf-8") as f:
    record_lines = [line for line in f if line.strip()]
with open(full_path, "r", encoding="utf-8") as f:
    full_lines = [line for line in f if line.strip()]
if len(record_lines) != len(full_lines):
    raise SystemExit(
        f"incomplete cache: {len(record_lines)} records for {len(full_lines)} rows in {cache_dir}"
    )
if not record_lines:
    raise SystemExit(f"empty records cache: {records_path}")

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

for field in required[1:]:
    for position in first[field]:
        total = sum(int(count) for _, count in position)
        if total < expected_samples:
            raise SystemExit(
                f"cache field {field} has only {total} samples; expected at least {expected_samples}"
            )
PY
}

preflight_caches() {
  local length seed cache_dir message failures=0
  for length in $LENGTHS; do
    for seed in $SEEDS; do
      cache_dir="$(cache_dir_for "$length" "$seed")"
      if ! message="$(check_cache_complete "$cache_dir" 2>&1)"; then
        echo "[CACHE ERROR] length=${length} seed=${seed}: ${message}" >&2
        failures=$((failures + 1))
      fi
    done
  done
  if (( failures > 0 )); then
    echo "Refusing to run: ${failures} required cache(s) are missing or incomplete." >&2
    exit 2
  fi
}

append_run_row() {
  local status="$1"
  local execution="$2"
  local length="$3"
  local seed="$4"
  local sample_count="$5"
  local result_name="$6"
  local cache_dir="$7"
  local roc_path="$8"
  local score_path="$9"
  local log_file="${10}"

  python - "$CSV_PATH" "$status" "$execution" "$length" "$seed" \
    "$sample_count" "$MODEL" "$DATA" "$NUM_SHOTS" "$CACHE_NUM_SAMPLES" \
    "$PREFIX_RATIO" "$FIXED_PREFIX_SEED" "$WPMIA_TAU" "$WPMIA_GAMMA" \
    "$result_name" "$cache_dir" "$roc_path" "$score_path" "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    execution,
    length,
    seed,
    sample_count,
    model,
    data,
    num_shots,
    cache_num_samples,
    prefix_ratio,
    prefix_seed,
    tau,
    gamma,
    result_name,
    cache_dir,
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

def metrics_from_scores(path):
    import numpy as np
    from sklearn.metrics import auc as auc_fn
    from sklearn.metrics import roc_curve

    labels = []
    scores = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float))
    accuracy = np.max(1 - (fpr + (1 - tpr)) / 2)

    def low_at(threshold):
        idx = np.where(fpr < threshold)[0]
        return float(tpr[idx[-1]]) if len(idx) else 0.0

    return {
        "auc_pct": f"{float(auc_fn(fpr, tpr)) * 100:.6f}",
        "accuracy_pct": f"{float(accuracy) * 100:.6f}",
        "tpr1_fpr_pct": f"{low_at(0.01) * 100:.6f}",
        "tpr5_fpr_pct": f"{low_at(0.05) * 100:.6f}",
        "tpr10_fpr_pct": f"{low_at(0.10) * 100:.6f}",
    }

if status == "ok" and os.path.exists(score_path):
    metrics = metrics_from_scores(score_path)
elif os.path.exists(log_file):
    pattern = re.compile(
        r"AUC (?P<auc>[0-9.]+), Accuracy (?P<acc>[0-9.]+), "
        r"TPR@1%FPR of (?P<tpr1>[0-9.]+), TPR@5%FPR of (?P<tpr5>[0-9.]+), "
        r"TPR@10%FPR of (?P<tpr10>[0-9.]+)"
    )
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
    "status",
    "execution",
    "method",
    "length",
    "sample_count",
    "sampling_seed",
    "fixed_prefix_seed",
    "run_count",
    "model",
    "data",
    "sub_dataset",
    "num_shots",
    "cache_num_samples",
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
    "auc_variance_pct2",
    "auc_min_pct",
    "auc_max_pct",
    "accuracy_mean_pct",
    "accuracy_std_pct",
    "tpr5_fpr_mean_pct",
    "tpr5_fpr_std_pct",
    "result_name",
    "cache_dir",
    "roc_path",
    "score_path",
    "log_file",
]

row = {
    "row_type": "run",
    "status": status,
    "execution": execution,
    "method": "wpmia",
    "length": length,
    "sample_count": sample_count,
    "sampling_seed": seed,
    "fixed_prefix_seed": prefix_seed,
    "run_count": "",
    "model": model,
    "data": data,
    "sub_dataset": f"WikiMIA_length{length}",
    "num_shots": num_shots,
    "cache_num_samples": cache_num_samples,
    "prefix_ratio": prefix_ratio,
    "wpmia_tau": tau,
    "wpmia_gamma": gamma,
    "result_name": result_name,
    "cache_dir": cache_dir,
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
  python - "$CSV_PATH" "$SUMMARY_CSV_PATH" "$STD_DDOF" "$SEEDS" <<'PY'
import csv
import os
import statistics
import sys
from collections import defaultdict

csv_path, summary_path, std_ddof, seed_text = sys.argv[1:]
std_ddof = int(std_ddof)
expected_runs = len(seed_text.split())
if not os.path.exists(csv_path):
    raise SystemExit(0)

with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if row.get("row_type") != "summary"]

# Keep the latest row for a rerun of the same configuration.
deduplicated = {}
for row in rows:
    key = (row["length"], row["sample_count"], row["sampling_seed"])
    deduplicated[key] = row
rows = sorted(
    deduplicated.values(),
    key=lambda row: (int(row["length"]), int(row["sample_count"]), int(row["sampling_seed"])),
)

groups = defaultdict(list)
for row in rows:
    if row.get("status") == "ok" and row.get("auc_pct"):
        groups[(row["length"], row["sample_count"])].append(row)

summary_rows = []
for (length, sample_count), group in sorted(
    groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
):
    base = group[0]
    auc_values = [float(row["auc_pct"]) for row in group]
    accuracy_values = [float(row["accuracy_pct"]) for row in group]
    tpr5_values = [float(row["tpr5_fpr_pct"]) for row in group]

    def sample_std(values):
        if len(values) <= 1:
            return 0.0
        return statistics.pstdev(values) if std_ddof == 0 else statistics.stdev(values)

    auc_std = sample_std(auc_values)
    summary = {field: "" for field in fieldnames}
    summary.update(
        {
            "row_type": "summary",
            "status": "ok" if len(group) == expected_runs else "incomplete",
            "execution": "aggregate",
            "method": "wpmia",
            "length": length,
            "sample_count": sample_count,
            "fixed_prefix_seed": base["fixed_prefix_seed"],
            "run_count": str(len(group)),
            "model": base["model"],
            "data": base["data"],
            "sub_dataset": base["sub_dataset"],
            "num_shots": base["num_shots"],
            "cache_num_samples": base["cache_num_samples"],
            "prefix_ratio": base["prefix_ratio"],
            "wpmia_tau": base["wpmia_tau"],
            "wpmia_gamma": base["wpmia_gamma"],
            "auc_mean_pct": f"{statistics.fmean(auc_values):.6f}",
            "auc_std_pct": f"{auc_std:.6f}",
            "auc_variance_pct2": f"{auc_std ** 2:.6f}",
            "auc_min_pct": f"{min(auc_values):.6f}",
            "auc_max_pct": f"{max(auc_values):.6f}",
            "accuracy_mean_pct": f"{statistics.fmean(accuracy_values):.6f}",
            "accuracy_std_pct": f"{sample_std(accuracy_values):.6f}",
            "tpr5_fpr_mean_pct": f"{statistics.fmean(tpr5_values):.6f}",
            "tpr5_fpr_std_pct": f"{sample_std(tpr5_values):.6f}",
        }
    )
    summary_rows.append(summary)

with open(csv_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    writer.writerows(summary_rows)

with open(summary_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)
PY
}

score_dump_valid() {
  local score_path="$1"
  python - "$score_path" <<'PY'
import csv
import os
import sys

path = sys.argv[1]
if not os.path.exists(path) or os.path.getsize(path) == 0:
    raise SystemExit(1)
with open(path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows or not {"index", "label", "score"}.issubset(rows[0]):
    raise SystemExit(1)
PY
}

run_one() {
  local length="$1"
  local seed="$2"
  local sample_count="$3"
  local sub_dataset="WikiMIA_length${length}"
  local cache_root_for_seed="${CACHE_ROOT}/sampling_seed_${seed}"
  local cache_dir result_name roc_source roc_path score_path log_file status execution
  local tau_slug gamma_slug

  cache_dir="$(cache_dir_for "$length" "$seed")"
  tau_slug="$(format_param_slug "$WPMIA_TAU")"
  gamma_slug="$(format_param_slug "$WPMIA_GAMMA")"
  result_name="wpmia_rebuttal_sampling_budget_len${length}_m${sample_count}_seed${seed}_tau_${tau_slug}_gamma_${gamma_slug}"
  roc_source="${cache_dir}/${result_name}_roc_tpr_at_5_fpr.png"
  roc_path="${ROC_ROOT}/len${length}/m${sample_count}/seed${seed}.png"
  score_path="${SCORE_ROOT}/len${length}_m${sample_count}_seed${seed}.csv"
  log_file="${RUN_ROOT}/len${length}_m${sample_count}_seed${seed}.log"

  mkdir -p "$(dirname "$roc_path")"
  status="ok"
  execution="computed"

  if [[ "$SKIP_COMPLETED" == "1" ]] && score_dump_valid "$score_path"; then
    execution="reused_score"
    echo "[REUSE] length=${length} M=${sample_count} seed=${seed}"
  else
    read -r -a simmia_cmd <<< "$SIMMIA_BIN"
    cmd=(
      "${simmia_cmd[@]}"
      --model_name_or_path "$MODEL"
      --sampling relative_word_by_word
      --postprocess process_wpmia_word_data
      --inference wpmia_score
      --output_dir "$cache_root_for_seed"
      --result_name "$result_name"
      --score_dump_path "$score_path"
      --num_samples "$CACHE_NUM_SAMPLES"
      --sample_count_limit "$sample_count"
      --data "$DATA"
      --sub_dataset "$sub_dataset"
      --num_shots "$NUM_SHOTS"
      --prefix_ratio "$PREFIX_RATIO"
      --top_k "$TOP_K"
      --seed "$seed"
      --prefix_seed "$FIXED_PREFIX_SEED"
      --sampling_seed "$seed"
      --wpmia_tau "$WPMIA_TAU"
      --wpmia_gamma "$WPMIA_GAMMA"
    )
    if [[ -n "${GPU_IDS:-}" ]]; then
      local gpu_ids_norm
      gpu_ids_norm="${GPU_IDS//,/ }"
      read -r -a gpu_id_arr <<< "$gpu_ids_norm"
      cmd+=(--gpu_ids "${gpu_id_arr[@]}")
    fi
    [[ -n "${CONCURRENCY:-}" ]] && cmd+=(--concurrency "$CONCURRENCY")
    cmd+=("${EXTRA_ARGS[@]}")

    echo "[START] length=${length} M=${sample_count} seed=${seed}"
    {
      echo "COMMAND: ${cmd[*]}"
      echo
    } > "$log_file"

    if [[ "$DRY_RUN" == "1" ]]; then
      status="dry_run"
      execution="dry_run"
      echo "[DRY_RUN] ${cmd[*]}" | tee -a "$log_file"
    elif ! "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
      status="failed"
    fi
  fi

  if [[ "$status" == "ok" && -f "$roc_source" ]]; then
    cp "$roc_source" "$roc_path"
  fi

  append_run_row "$status" "$execution" "$length" "$seed" "$sample_count" \
    "$result_name" "$cache_dir" "$roc_path" "$score_path" "$log_file"

  echo "[DONE] length=${length} M=${sample_count} seed=${seed} status=${status}"
  if [[ "$status" == "failed" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    exit 1
  fi
}

echo "Preflight: validating all required five-seed caches..."
preflight_caches
echo "Preflight passed. No target-LLM sampling will be performed."
echo "Lengths: $LENGTHS"
echo "Sampling seeds: $SEEDS"
echo "Sample budgets: $SAMPLE_COUNTS"
echo "WPMIA tau/gamma: $WPMIA_TAU / $WPMIA_GAMMA"
echo "Run logs: $RUN_ROOT"

for length in $LENGTHS; do
  for sample_count in $SAMPLE_COUNTS; do
    for seed in $SEEDS; do
      run_one "$length" "$seed" "$sample_count"
    done
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  write_summary_rows
fi

echo "Finished sampling-budget stability experiment."
echo "All rows: $CSV_PATH"
echo "Summary rows: $SUMMARY_CSV_PATH"
echo "ROC copies: $ROC_ROOT"
