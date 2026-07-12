#!/usr/bin/env bash
set -euo pipefail

# Queue paper-style black-box MIA runs and generate the expensive sampling caches.
#
# The command contract follows README.md's "Reproducing Paper Results" section:
#   bash scripts/run_samia.sh <MODEL> <DATA> <SUB_DATASET> [GPU_IDS] [CONCURRENCY]
#   bash scripts/run_simmia_hard.sh <MODEL> <DATA> <SUB_DATASET> [GPU_IDS] [CONCURRENCY]
#   bash scripts/run_simmia_soft.sh <MODEL> <DATA> <SUB_DATASET> [GPU_IDS] [CONCURRENCY]
#
# This script is meant for long background runs that gradually reproduce the
# SimMIA / SimMIA* rows in the paper tables while also materializing the new
# records.jsonl cache format:
#   - sample_results
#   - nonmember_prefix_sample_results
#   - member_prefix_sample_results
#
# It runs tasks sequentially so only one model/dataset job is active at a time.
# Recommended background usage:
#
#   mkdir -p logs
#   nohup bash scripts/run_paper_sampling_cache_queue.sh "0 1 2 3 4 5 6 7" \
#     > logs/simmia_cache_queue.nohup.log 2>&1 &
#
# Useful controls:
#   METHODS="hard soft"      # hard=SimMIA*, soft=SimMIA; add wpmia to score WPMIA
#   RUN_WIKIMIA=1            # WikiMIA table workloads
#   RUN_MIMIR=1              # MIMIR table workloads
#   RUN_WIKIMIA25=1          # WikiMIA-25 open-model workloads
#   RUN_SAMIA=0              # optionally run SaMIA baseline into samia_out
#   RUN_API=0                # optionally include API models listed in API_WIKIMIA25_MODELS
#   CLEAN=0                  # set to 1 to delete each target simmia_out cache dir first
#   CONTINUE_ON_ERROR=0      # set to 1 to keep queue running after a failed task
#   MIMIR_SPLIT=ngram_7_0.2  # passed to simmia.benchmark as --split for MIMIR
#
# Notes:
#   - hard and soft SimMIA share the same records.jsonl.
#   - If one method fails before records.jsonl is complete, later methods for
#     the same model/dataset are skipped so they do not continue target-LLM
#     sampling under a different log name.
#   - hard writes simmia_hard_roc_tpr_at_5_fpr.png; soft writes
#     simmia_roc_tpr_at_5_fpr.png in the same output folder.
#   - If an old-format records.jsonl exists, this script stops and asks you to
#     rerun with CLEAN=1, because old caches do not contain the new explicit
#     nonmember/member prefix sampling fields.
#   - Some model ids, especially LLaMA, Qwen, and API model names, may differ
#     in your local/HF/API setup. Override the *_MODELS variables below when
#     needed. README.md explicitly shows api:google/gemini-2.5-flash as the API
#     example; Anthropic/OpenAI ids should be supplied from your available API.

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-}}"
EXTRA_ARGS=("${@:3}")

METHODS="${METHODS:-hard soft}"
RUN_WIKIMIA="${RUN_WIKIMIA:-1}"
RUN_MIMIR="${RUN_MIMIR:-1}"
RUN_WIKIMIA25="${RUN_WIKIMIA25:-1}"
RUN_SAMIA="${RUN_SAMIA:-0}"
RUN_API="${RUN_API:-0}"
CLEAN="${CLEAN:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

MIMIR_SPLIT="${MIMIR_SPLIT:-ngram_7_0.2}"

# Table 1: WikiMIA. Override WIKIMIA_MODELS if your LLaMA id differs.
WIKIMIA_LENGTHS="${WIKIMIA_LENGTHS:-32 64 128}"
WIKIMIA_MODELS="${WIKIMIA_MODELS:-facebook/opt-6.7b EleutherAI/pythia-6.9b huggyllama/llama-13b EleutherAI/gpt-neox-20b}"

# Table 2: MIMIR.
MIMIR_SUBSETS="${MIMIR_SUBSETS:-wikipedia_(en) github pile_cc pubmed_central arxiv dm_mathematics hackernews}"
MIMIR_MODELS="${MIMIR_MODELS:-EleutherAI/pythia-160m EleutherAI/pythia-1.4b EleutherAI/pythia-2.8b EleutherAI/pythia-6.9b}"

# Table 3: WikiMIA-25. API models are opt-in via RUN_API=1.
WIKIMIA25_SUBSETS="${WIKIMIA25_SUBSETS:-paper_subset}"
WIKIMIA25_MODELS="${WIKIMIA25_MODELS-EleutherAI/pythia-6.9b Qwen/Qwen3-8B-Base}"
API_WIKIMIA25_MODELS="${API_WIKIMIA25_MODELS:-api:google/gemini-2.5-flash}"

RUN_ROOT="${RUN_ROOT:-logs/simmia_cache_queue/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"
SUMMARY_LOG="$RUN_ROOT/summary.log"

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

normalize_wikimia_subdataset() {
  local sub="$1"
  if [[ "$sub" =~ ^[0-9]+$ ]]; then
    printf 'WikiMIA_length%s' "$sub"
  else
    printf '%s' "$sub"
  fi
}

cache_dir_for() {
  local data="$1"
  local sub="$2"
  local model="$3"
  local data_base="${data##*/}"
  local model_base="${model##*/}"

  if [[ "${data,,}" == *mimir* ]]; then
    printf 'simmia_out/%s/%s/%s/%s/10' "$data_base" "$MIMIR_SPLIT" "$model_base" "$sub"
  else
    printf 'simmia_out/%s/%s/%s/7' "$data_base" "$sub" "$model_base"
  fi
}

check_new_records_shape() {
  local cache_dir="$1"
  local records="$cache_dir/records.jsonl"
  [[ -f "$records" ]] || return 0

  python - "$records" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    line = f.readline()
if not line.strip():
    sys.exit(0)
record = json.loads(line)
required = {
    "sample_results",
    "nonmember_prefix_sample_results",
    "member_prefix_sample_results",
}
missing = sorted(required - set(record))
if missing:
    print(f"old_or_incomplete_records={path}")
    print("missing=" + ",".join(missing))
    sys.exit(2)
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

if not os.path.exists(records_path) or not os.path.exists(full_path):
    sys.exit(2)

with open(records_path, "r", encoding="utf-8") as f:
    first = f.readline()
if not first.strip():
    sys.exit(2)

record = json.loads(first)
if required - set(record):
    sys.exit(2)

with open(records_path, "rb") as f:
    records_count = sum(1 for _ in f)
with open(full_path, "rb") as f:
    full_count = sum(1 for _ in f)

if records_count != full_count:
    sys.exit(2)
PY
}

build_wrapper_cmd() {
  local method="$1"
  local model="$2"
  local data="$3"
  local sub="$4"
  local script=""

  case "$method" in
    hard) script="scripts/run_simmia_hard.sh" ;;
    soft) script="scripts/run_simmia_soft.sh" ;;
    wpmia) script="scripts/run_wpmia.sh" ;;
    samia) script="scripts/run_samia.sh" ;;
    *)
      echo "Unknown method: $method" >&2
      return 2
      ;;
  esac

  WRAPPER_CMD=(bash "$script" "$model" "$data" "$sub" "$GPU_IDS")
  if [[ -n "$CONCURRENCY" || ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    WRAPPER_CMD+=("$CONCURRENCY")
  fi
  if [[ "${data,,}" == *mimir* ]]; then
    WRAPPER_CMD+=(--split "$MIMIR_SPLIT")
  fi
  WRAPPER_CMD+=("${EXTRA_ARGS[@]}")
}

run_command() {
  local label="$1"
  shift
  local log_file="$RUN_ROOT/$(slugify "$label").log"

  log "START $label"
  log "LOG   $log_file"
  printf 'COMMAND: %q' "$@" > "$log_file"
  printf '\n\n' >> "$log_file"

  if "$@" 2>&1 | tee -a "$log_file"; then
    log "DONE  $label"
    return 0
  else
    local status=$?
    log "FAIL  $label (exit $status)"
    return "$status"
  fi
}

prepare_cache_dir() {
  local cache_dir="$1"
  if [[ "$CLEAN" == "1" ]]; then
    log "CLEAN $cache_dir"
    rm -rf "$cache_dir"
  else
    if ! check_new_records_shape "$cache_dir"; then
      log "ERROR old-format records detected under $cache_dir"
      log "Rerun with CLEAN=1 to regenerate this cache with explicit member/nonmember fields."
      exit 2
    fi
  fi
}

run_simmia_combo() {
  local model="$1"
  local data="$2"
  local sub="$3"
  local cache_dir="$4"

  prepare_cache_dir "$cache_dir"

  for method in $METHODS; do
    build_wrapper_cmd "$method" "$model" "$data" "$sub"
    if run_command "$method--$data--$sub--$model" "${WRAPPER_CMD[@]}"; then
      continue
    else
      local status=$?
      if ! check_cache_complete "$cache_dir"; then
        log "SKIP remaining methods for $data--$sub--$model because records.jsonl is incomplete."
        if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
          return 0
        fi
        return "$status"
      fi

      log "CACHE complete for $data--$sub--$model after $method failed."
      if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
        log "CONTINUE with remaining methods for this complete cache."
        continue
      fi
      return "$status"
    fi
  done
}

run_samia_combo() {
  local model="$1"
  local data="$2"
  local sub="$3"
  [[ "$RUN_SAMIA" == "1" ]] || return 0
  build_wrapper_cmd samia "$model" "$data" "$sub"
  if run_command "samia--$data--$sub--$model" "${WRAPPER_CMD[@]}"; then
    return 0
  else
    local status=$?
    if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
      return 0
    fi
    return "$status"
  fi
}

log "Run root: $RUN_ROOT"
log "GPU_IDS: $GPU_IDS"
log "METHODS: $METHODS"
log "CLEAN: $CLEAN"

if [[ "$RUN_WIKIMIA" == "1" ]]; then
  for model in $WIKIMIA_MODELS; do
    for len in $WIKIMIA_LENGTHS; do
      sub="$(normalize_wikimia_subdataset "$len")"
      cache_dir="$(cache_dir_for "swj0419/WikiMIA" "$sub" "$model")"
      run_simmia_combo "$model" "swj0419/WikiMIA" "$sub" "$cache_dir"
      run_samia_combo "$model" "swj0419/WikiMIA" "$sub"
    done
  done
fi

if [[ "$RUN_MIMIR" == "1" ]]; then
  for model in $MIMIR_MODELS; do
    for sub in $MIMIR_SUBSETS; do
      cache_dir="$(cache_dir_for "iamgroot42/mimir" "$sub" "$model")"
      run_simmia_combo "$model" "iamgroot42/mimir" "$sub" "$cache_dir"
      run_samia_combo "$model" "iamgroot42/mimir" "$sub"
    done
  done
fi

if [[ "$RUN_WIKIMIA25" == "1" ]]; then
  wikimia25_models="$WIKIMIA25_MODELS"
  if [[ "$RUN_API" == "1" ]]; then
    wikimia25_models="$wikimia25_models $API_WIKIMIA25_MODELS"
  fi

  for model in $wikimia25_models; do
    for sub in $WIKIMIA25_SUBSETS; do
      cache_dir="$(cache_dir_for "SimMIA/WikiMIA-25" "$sub" "$model")"
      run_simmia_combo "$model" "SimMIA/WikiMIA-25" "$sub" "$cache_dir"
      run_samia_combo "$model" "SimMIA/WikiMIA-25" "$sub"
    done
  done
fi

log "All queued tasks finished."
