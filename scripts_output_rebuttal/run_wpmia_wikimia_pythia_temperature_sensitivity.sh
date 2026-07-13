#!/usr/bin/env bash
set -euo pipefail

# Rebuttal experiment: WPMIA temperature sensitivity on WikiMIA/Pythia-6.9B.
#
# The target-model cache is reused from simmia_out. Only lightweight WPMIA
# postprocessing and inference are run for tau=0.10,...,0.50 (step 0.05).
# All generated files stay under scripts_output_rebuttal.
#
# Full run:
#   nohup bash \
#     scripts_output_rebuttal/run_wpmia_wikimia_pythia_temperature_sensitivity.sh \
#     "0 1 2 3 4 5 6 7" \
#     > scripts_output_rebuttal/wikimia_pythia69b_temperature.nohup.log 2>&1 &
#
# Small validation run:
#   DRY_RUN=1 LENGTHS="32" TAUS="0.10 0.15" bash \
#     scripts_output_rebuttal/run_wpmia_wikimia_pythia_temperature_sensitivity.sh "0"

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

MODEL="${MODEL:-EleutherAI/pythia-6.9b}"
DATA="${DATA:-swj0419/WikiMIA}"
LENGTHS="${LENGTHS:-32 64 128}"
TAUS="${TAUS:-0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
NUM_SHOTS="${NUM_SHOTS:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
PREFIX_RATIO="${PREFIX_RATIO:-0.0}"
TOP_K="${TOP_K:-20}"
SEED="${SEED:-42}"
SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-simmia_out/WikiMIA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-scripts_output_rebuttal}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${OUTPUT_ROOT}/temperature_sensitivity}"
CACHE_VIEW_ROOT="${CACHE_VIEW_ROOT:-${EXPERIMENT_ROOT}/cache_view}"
RUN_ROOT="${RUN_ROOT:-${EXPERIMENT_ROOT}/logs/$(date +%Y%m%d_%H%M%S)}"
SCORE_ROOT="${SCORE_ROOT:-${EXPERIMENT_ROOT}/scores}"
ROC_ROOT="${ROC_ROOT:-${EXPERIMENT_ROOT}/roc}"
TABLE_CSV="${TABLE_CSV:-${OUTPUT_ROOT}/wikimia_pythia69b_wpmia_temperature_auc.csv}"
DETAIL_CSV="${DETAIL_CSV:-${EXPERIMENT_ROOT}/wikimia_pythia69b_wpmia_temperature_details.csv}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -n "${SIMMIA_BIN:-}" ]]; then
  read -r -a SIMMIA_CMD <<< "$SIMMIA_BIN"
elif command -v simmia.benchmark >/dev/null 2>&1; then
  SIMMIA_CMD=(simmia.benchmark)
elif [[ -x "$HOME/miniconda3/envs/simmia/bin/simmia.benchmark" ]]; then
  SIMMIA_CMD=("$HOME/miniconda3/envs/simmia/bin/simmia.benchmark")
else
  SIMMIA_CMD=(simmia.benchmark)
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ "${SIMMIA_CMD[0]}" == */* && -x "$(dirname "${SIMMIA_CMD[0]}")/python" ]]; then
  PYTHON_CMD="$(dirname "${SIMMIA_CMD[0]}")/python"
elif [[ -x "$HOME/miniconda3/envs/simmia/bin/python" ]]; then
  PYTHON_CMD="$HOME/miniconda3/envs/simmia/bin/python"
else
  PYTHON_CMD=python
fi

mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$SCORE_ROOT" "$ROC_ROOT" "$CACHE_VIEW_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-simmia}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$MPLCONFIGDIR"

format_param_slug() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/_neg_}"
  printf '%s' "$value"
}

source_cache_dir() {
  local length="$1"
  printf '%s/WikiMIA_length%s/%s/%s' \
    "$SOURCE_CACHE_ROOT" "$length" "${MODEL##*/}" "$NUM_SHOTS"
}

view_cache_dir() {
  local length="$1"
  printf '%s/WikiMIA/WikiMIA_length%s/%s/%s' \
    "$CACHE_VIEW_ROOT" "$length" "${MODEL##*/}" "$NUM_SHOTS"
}

check_cache_complete() {
  local cache_dir="$1"
  "$PYTHON_CMD" - "$cache_dir" "$NUM_SAMPLES" <<'PY'
import json
import os
import sys

cache_dir, expected_samples = sys.argv[1:]
expected_samples = int(expected_samples)
records_path = os.path.join(cache_dir, "records.jsonl")
full_path = os.path.join(cache_dir, "full_dataset.jsonl")
prefix_path = os.path.join(cache_dir, "prefix_data.json")

for path in (records_path, full_path, prefix_path):
    if not os.path.isfile(path):
        raise SystemExit(f"missing cache file: {path}")

with open(records_path, "r", encoding="utf-8") as f:
    record_lines = [line for line in f if line.strip()]
with open(full_path, "r", encoding="utf-8") as f:
    full_lines = [line for line in f if line.strip()]
if not record_lines or len(record_lines) != len(full_lines):
    raise SystemExit(
        f"incomplete cache: records={len(record_lines)}, full_dataset={len(full_lines)}"
    )

required = (
    "label_results",
    "sample_results",
    "nonmember_prefix_sample_results",
    "member_prefix_sample_results",
)
record = json.loads(record_lines[0])
missing = [field for field in required if field not in record]
if missing:
    raise SystemExit("cache is missing WPMIA fields: " + ", ".join(missing))

for field in required[1:]:
    for position in record[field]:
        count = sum(int(value) for _, value in position)
        if count < expected_samples:
            raise SystemExit(
                f"{field} contains only {count} samples; expected {expected_samples}"
            )
PY
}

prepare_cache_view() {
  local length="$1"
  local source_dir view_dir file
  source_dir="$(source_cache_dir "$length")"
  view_dir="$(view_cache_dir "$length")"
  check_cache_complete "$source_dir"
  mkdir -p "$view_dir"
  for file in records.jsonl full_dataset.jsonl prefix_data.json; do
    ln -sfn "$(realpath "$source_dir/$file")" "$view_dir/$file"
  done
}

score_dump_valid() {
  local path="$1"
  "$PYTHON_CMD" - "$path" <<'PY'
import csv
import os
import sys

path = sys.argv[1]
if not os.path.isfile(path) or os.path.getsize(path) == 0:
    raise SystemExit(1)
with open(path, "r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
if not rows or not {"index", "label", "score"}.issubset(rows[0]):
    raise SystemExit(1)
if len({row["label"] for row in rows}) != 2:
    raise SystemExit(1)
PY
}

run_one() {
  local length="$1"
  local tau="$2"
  local tau_slug result_name cache_dir score_path roc_source roc_path log_file status
  tau_slug="$(format_param_slug "$tau")"
  result_name="wpmia_rebuttal_temperature_len${length}_tau_${tau_slug}_gamma_1p0"
  cache_dir="$(view_cache_dir "$length")"
  score_path="${SCORE_ROOT}/len${length}_tau_${tau_slug}.csv"
  roc_source="${cache_dir}/${result_name}_roc_tpr_at_5_fpr.png"
  roc_path="${ROC_ROOT}/len${length}_tau_${tau_slug}.png"
  log_file="${RUN_ROOT}/len${length}_tau_${tau_slug}.log"

  if [[ "$SKIP_COMPLETED" == "1" ]] && score_dump_valid "$score_path"; then
    echo "[REUSE] length=${length} tau=${tau}"
    return 0
  fi

  cmd=(
    "${SIMMIA_CMD[@]}"
    --model_name_or_path "$MODEL"
    --sampling relative_word_by_word
    --postprocess process_wpmia_word_data
    --inference wpmia_score
    --output_dir "$CACHE_VIEW_ROOT"
    --result_name "$result_name"
    --score_dump_path "$score_path"
    --num_samples "$NUM_SAMPLES"
    --data "$DATA"
    --sub_dataset "WikiMIA_length${length}"
    --num_shots "$NUM_SHOTS"
    --prefix_ratio "$PREFIX_RATIO"
    --top_k "$TOP_K"
    --seed "$SEED"
    --prefix_seed "$SEED"
    --sampling_seed "$SEED"
    --wpmia_tau "$tau"
    --wpmia_gamma "$WPMIA_GAMMA"
  )
  if [[ -n "$GPU_IDS" ]]; then
    local normalized_gpu_ids
    local -a gpu_id_array=()
    normalized_gpu_ids="${GPU_IDS//,/ }"
    read -r -a gpu_id_array <<< "$normalized_gpu_ids"
    cmd+=(--gpu_ids "${gpu_id_array[@]}")
  fi
  [[ -n "$CONCURRENCY" ]] && cmd+=(--concurrency "$CONCURRENCY")
  cmd+=("${EXTRA_ARGS[@]}")

  echo "[START] length=${length} tau=${tau}"
  {
    printf 'COMMAND:'
    printf ' %q' "${cmd[@]}"
    printf '\n\n'
  } > "$log_file"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] ${cmd[*]}" | tee -a "$log_file"
    return 0
  fi

  status="ok"
  if ! "${cmd[@]}" >> "$log_file" 2>&1; then
    status="failed"
  fi
  if [[ "$status" == "ok" ]] && score_dump_valid "$score_path"; then
    [[ -f "$roc_source" ]] && cp "$roc_source" "$roc_path"
  else
    echo "[FAILED] length=${length} tau=${tau}; see $log_file" >&2
    if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
      exit 1
    fi
  fi
  echo "[DONE] length=${length} tau=${tau} status=${status}"
}

write_tables() {
  "$PYTHON_CMD" - "$SCORE_ROOT" "$DETAIL_CSV" "$TABLE_CSV" "$LENGTHS" "$TAUS" \
    "$MODEL" "$SEED" "$WPMIA_GAMMA" <<'PY'
import csv
import os
import sys

import numpy as np
from sklearn.metrics import auc, roc_curve

score_root, detail_path, table_path, lengths_text, taus_text, model, seed, gamma = sys.argv[1:]
lengths = lengths_text.split()
taus = taus_text.split()

def slug(value):
    return value.replace(".", "p").replace("-", "_neg_")

def score_auc(path):
    labels = []
    scores = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
            scores.append(float(row["score"]))
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=bool), np.asarray(scores))
    return float(auc(fpr, tpr) * 100.0), len(labels)

details = []
auc_by_setting = {}
for tau in taus:
    for length in lengths:
        path = os.path.join(score_root, f"len{length}_tau_{slug(tau)}.csv")
        if not os.path.isfile(path):
            raise SystemExit(f"missing score dump: {path}")
        auc_pct, examples = score_auc(path)
        auc_by_setting[(tau, length)] = auc_pct
        details.append(
            {
                "tau": tau,
                "length": length,
                "auc_pct": f"{auc_pct:.6f}",
                "num_examples": examples,
                "model": model,
                "seed": seed,
                "gamma": gamma,
                "score_path": path,
            }
        )

os.makedirs(os.path.dirname(detail_path), exist_ok=True)
with open(detail_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=details[0].keys())
    writer.writeheader()
    writer.writerows(details)

table_fields = ["tau"] + [f"len{length}_auc_pct" for length in lengths]
table_rows = []
for tau in taus:
    row = {"tau": tau}
    for length in lengths:
        row[f"len{length}_auc_pct"] = f"{auc_by_setting[(tau, length)]:.2f}"
    table_rows.append(row)

os.makedirs(os.path.dirname(table_path), exist_ok=True)
with open(table_path, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=table_fields)
    writer.writeheader()
    writer.writerows(table_rows)
PY
}

echo "Preflight: validating default WikiMIA/Pythia-6.9B caches..."
for length in $LENGTHS; do
  prepare_cache_view "$length"
done
echo "Preflight passed. Target-model sampling will not run."
echo "Lengths: $LENGTHS"
echo "Tau values: $TAUS"
echo "Fixed config: seed=$SEED gamma=$WPMIA_GAMMA shots=$NUM_SHOTS samples=$NUM_SAMPLES prefix_ratio=$PREFIX_RATIO"
echo "Run logs: $RUN_ROOT"

for tau in $TAUS; do
  for length in $LENGTHS; do
    run_one "$length" "$tau"
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  write_tables
  echo "Finished temperature-sensitivity experiment."
  echo "AUC table: $TABLE_CSV"
  echo "Detailed results: $DETAIL_CSV"
  echo "ROC copies: $ROC_ROOT"
fi
