#!/usr/bin/env bash
set -euo pipefail

# Sweep WPMIA tau/gamma over WikiMIA-25 API caches. This is a thin wrapper over
# run_wpmia_wikimia25_cache_sweep.sh that maps api:openai/<model-id> to the
# cache directory name used by simmia.run (the final path segment of the model).
#
# Usage:
#   API_WIKIMIA25_MODELS="api:openai/claude-4-5-haiku api:openai/gemini-2.5-flash api:openai/gpt-5-chat-latest" \
#   nohup bash scripts/run_wpmia_wikimia25_api_sweep.sh "0 1 2 3 4 5 6 7" 5 \
#     > logs/wpmia/wikimia25_api_wpmia_sweep.nohup.log 2>&1 &
#
# Set PARALLEL=1 to launch one sweep process per model/subset.

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-5}}"
EXTRA_ARGS=("${@:3}")

API_WIKIMIA25_MODELS="${API_WIKIMIA25_MODELS:-api:openai/claude-4-5-haiku api:openai/gemini-2.5-flash api:openai/gpt-5-chat-latest}"
WIKIMIA25_SUBSETS="${WIKIMIA25_SUBSETS:-paper_subset}"
NUM_SHOTS="${NUM_SHOTS:-7}"
CSV_ROOT="${CSV_ROOT:-logs/wpmia}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-logs/wpmia}"
PARALLEL="${PARALLEL:-0}"

mkdir -p "$CSV_ROOT" "$RUN_ROOT_BASE"

slugify() {
  local s="$1"
  s="${s#api:}"
  s="${s//\//__}"
  s="${s//[^A-Za-z0-9._-]/_}"
  printf '%s' "$s"
}

model_dir_from_api_id() {
  local s="$1"
  s="${s#api:}"
  s="${s#openai/}"
  printf '%s' "${s##*/}"
}

run_one_cache() {
  local sub_dataset="$1"
  local model_id="$2"
  local model_dir
  local cache_root
  local run_root
  local log_file

  model_dir="$(model_dir_from_api_id "$model_id")"
  cache_root="simmia_out/WikiMIA-25/${sub_dataset}/${model_dir}/${NUM_SHOTS}"

  if [[ ! -d "$cache_root" ]]; then
    printf 'SKIP missing cache root: %s\n' "$cache_root" >&2
    return 0
  fi

  run_root="${RUN_ROOT_BASE}/$(date +%Y%m%d_%H%M%S)_wikimia25_$(slugify "$model_dir")_wpmia"
  log_file="${CSV_ROOT}/wikimia25_$(slugify "$model_dir")_wpmia_sweep.nohup.log"

  printf 'START WPMIA sweep: cache=%s log=%s\n' "$cache_root" "$log_file"
  CACHE_ROOT="$cache_root" \
  RUN_ROOT="$run_root" \
  CSV_ROOT="$CSV_ROOT" \
  bash scripts/run_wpmia_wikimia25_cache_sweep.sh \
    "$GPU_IDS" "$CONCURRENCY" \
    "${EXTRA_ARGS[@]}" \
    > "$log_file" 2>&1
}

pids=()
for sub_dataset in $WIKIMIA25_SUBSETS; do
  for model_id in $API_WIKIMIA25_MODELS; do
    if [[ "$PARALLEL" == "1" ]]; then
      run_one_cache "$sub_dataset" "$model_id" &
      pids+=("$!")
    else
      run_one_cache "$sub_dataset" "$model_id"
    fi
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
