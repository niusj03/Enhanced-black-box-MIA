#!/usr/bin/env bash
set -euo pipefail

# WPMIA embedding-model ablation on WikiMIA / Pythia-6.9B.
#
# Full run:
#   nohup bash scripts/run_wpmia_wikimia_pythia_embedding_ablation.sh "0 1 2 3 4 5 6 7" \
#     > logs/ablation/wikimia_pythia69b_embedding_wpmia.nohup.log 2>&1 &
#
# Small dry run:
#   DRY_RUN=1 LENGTHS="32" EMBEDDINGS="minilm" \
#   bash scripts/run_wpmia_wikimia_pythia_embedding_ablation.sh "0"

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

MODEL="${MODEL:-EleutherAI/pythia-6.9b}"
DATA="${DATA:-swj0419/WikiMIA}"
LENGTHS="${LENGTHS:-32 64 128}"
NUM_SHOTS="${NUM_SHOTS:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
PREFIX_RATIO="${PREFIX_RATIO:-0.0}"
WPMIA_TAU="${WPMIA_TAU:-0.2}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-simmia_out}"
SIMMIA_BIN="${SIMMIA_BIN:-simmia.benchmark}"
EMBEDDINGS="${EMBEDDINGS:-word2vec fasttext bge minilm uae mxbai}"
CSV_PATH="${CSV_PATH:-logs/ablation/wikimia_pythia69b_embedding_wpmia.csv}"
RUN_ROOT="${RUN_ROOT:-logs/ablation/$(date +%Y%m%d_%H%M%S)_wikimia_pythia69b_embedding_wpmia}"
CACHE_ONLY="${CACHE_ONLY:-1}"
DRY_RUN="${DRY_RUN:-0}"
RESET_CSV="${RESET_CSV:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "$RUN_ROOT" "$(dirname "$CSV_PATH")"
[[ "$RESET_CSV" == "1" ]] && rm -f "$CSV_PATH"

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

set_embedding_config() {
  local key="$1"
  case "$key" in
    word2vec)
      EMBEDDING_LABEL="Word2Vec"
      EMBEDDING_MODEL="gensim:word2vec-google-news-300"
      EMBEDDING_SLUG="word2vec"
      ;;
    fasttext)
      EMBEDDING_LABEL="fastText"
      EMBEDDING_MODEL="gensim:fasttext-wiki-news-subwords-300"
      EMBEDDING_SLUG="fasttext"
      ;;
    bge|bge-large-en-v1.5)
      EMBEDDING_LABEL="bge-large-en-v1.5"
      EMBEDDING_MODEL="BAAI/bge-large-en-v1.5"
      EMBEDDING_SLUG="bge_large_en_v1p5"
      ;;
    minilm|all-MiniLM-L6-v2)
      EMBEDDING_LABEL="all-MiniLM-L6-v2"
      EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
      EMBEDDING_SLUG="all_minilm_l6_v2"
      ;;
    uae|UAE-Large-V1)
      EMBEDDING_LABEL="UAE-Large-V1"
      EMBEDDING_MODEL="WhereIsAI/UAE-Large-V1"
      EMBEDDING_SLUG="uae_large_v1"
      ;;
    mxbai|mxbai-embed-large-v1)
      EMBEDDING_LABEL="mxbai-embed-large-v1"
      EMBEDDING_MODEL="mixedbread-ai/mxbai-embed-large-v1"
      EMBEDDING_SLUG="mxbai_embed_large_v1"
      ;;
    *)
      echo "Unknown embedding key '$key'. Use one of: word2vec fasttext bge minilm uae mxbai." >&2
      return 2
      ;;
  esac
}

append_csv_row() {
  local status="$1"
  local embedding_key="$2"
  local embedding_label="$3"
  local embedding_model="$4"
  local length="$5"
  local result_name="$6"
  local output_dir="$7"
  local roc_path="$8"
  local log_file="$9"

  python - "$CSV_PATH" "$status" "$embedding_key" "$embedding_label" \
    "$embedding_model" "$MODEL" "$DATA" "$length" "$NUM_SHOTS" "$NUM_SAMPLES" \
    "$PREFIX_RATIO" "$WPMIA_TAU" "$WPMIA_GAMMA" "$result_name" "$output_dir" \
    "$roc_path" "$log_file" <<'PY'
import csv
import os
import re
import sys

(
    csv_path,
    status,
    embedding_key,
    embedding_label,
    embedding_model,
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
    "embedding_key": embedding_key,
    "embedding_label": embedding_label,
    "embedding_model": embedding_model,
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
    "embedding_key",
    "embedding_label",
    "embedding_model",
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
    raise SystemExit(f"cache is missing required fields: {', '.join(missing)}")
PY
}

model_base="${MODEL##*/}"
data_base="${DATA##*/}"
tau_slug="$(format_param_slug "$WPMIA_TAU")"
gamma_slug="$(format_param_slug "$WPMIA_GAMMA")"

echo "Run root: $RUN_ROOT"
echo "CSV: $CSV_PATH"
echo "Model: $MODEL"
echo "Lengths: $LENGTHS"
echo "Embeddings: $EMBEDDINGS"
echo "WPMIA tau/gamma: $WPMIA_TAU / $WPMIA_GAMMA"
echo "Cache only: $CACHE_ONLY"

for embedding_key in $EMBEDDINGS; do
  set_embedding_config "$embedding_key"

  for length in $LENGTHS; do
    sub_dataset="WikiMIA_length${length}"
    output_dir="${OUTPUT_ROOT}/${data_base}/${sub_dataset}/${model_base}/${NUM_SHOTS}"
    result_name="wpmia_ablation_embedding_${EMBEDDING_SLUG}_len${length}_tau_${tau_slug}_gamma_${gamma_slug}"
    roc_path="${output_dir}/${result_name}_roc_tpr_at_5_fpr.png"
    log_file="${RUN_ROOT}/$(slugify "${sub_dataset}_${EMBEDDING_SLUG}").log"
    status="ok"

    if [[ "$CACHE_ONLY" == "1" ]]; then
      if ! cache_message="$(check_records_cache "$output_dir" 2>&1)"; then
        status="missing_cache"
        {
          echo "[SKIP] ${EMBEDDING_LABEL} length=${length}"
          echo "$cache_message"
        } | tee "$log_file"
        append_csv_row "$status" "$embedding_key" "$EMBEDDING_LABEL" "$EMBEDDING_MODEL" \
          "$length" "$result_name" "$output_dir" "$roc_path" "$log_file"
        continue
      fi
    fi

    read -r -a simmia_cmd <<< "$SIMMIA_BIN"
    cmd=(
      "${simmia_cmd[@]}"
      --model_name_or_path "$MODEL"
      --sampling relative_word_by_word
      --postprocess process_wpmia_word_data
      --inference wpmia_score
      --embedding_model "$EMBEDDING_MODEL"
      --output_dir "$OUTPUT_ROOT"
      --result_name "$result_name"
      --num_samples "$NUM_SAMPLES"
      --data "$DATA"
      --sub_dataset "$sub_dataset"
      --num_shots "$NUM_SHOTS"
      --prefix_ratio "$PREFIX_RATIO"
      --top_k 20
      --wpmia_tau "$WPMIA_TAU"
      --wpmia_gamma "$WPMIA_GAMMA"
    )

    if [[ -n "${GPU_IDS:-}" ]]; then
      gpu_ids_norm="${GPU_IDS//,/ }"
      read -r -a gpu_id_arr <<< "$gpu_ids_norm"
      if [[ "$EMBEDDING_MODEL" == gensim:* ]]; then
        cmd+=(--gpu_ids "${gpu_id_arr[0]}")
      else
        cmd+=(--gpu_ids "${gpu_id_arr[@]}")
      fi
    fi
    [[ -n "${CONCURRENCY:-}" ]] && cmd+=(--concurrency "$CONCURRENCY")
    cmd+=("${EXTRA_ARGS[@]}")

    echo "[START] embedding=${EMBEDDING_LABEL} length=${length}"
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

    append_csv_row "$status" "$embedding_key" "$EMBEDDING_LABEL" "$EMBEDDING_MODEL" \
      "$length" "$result_name" "$output_dir" "$roc_path" "$log_file"

    echo "[DONE] embedding=${EMBEDDING_LABEL} length=${length} status=${status}"
    if [[ "$status" == "failed" && "$CONTINUE_ON_ERROR" != "1" ]]; then
      echo "Stopping after failed run. Set CONTINUE_ON_ERROR=1 to keep sweeping." >&2
      exit 1
    fi
  done
done

echo "Finished. CSV written to $CSV_PATH"
