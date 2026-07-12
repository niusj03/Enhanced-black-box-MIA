#!/usr/bin/env bash
set -euo pipefail

# Queue WikiMIA-25 closed-source/API SimMIA* + SimMIA runs through an
# OpenAI-compatible endpoint. The API key is read from the environment and is
# never written by this script.
#
# Usage:
#   export OPENAI_API_KEY=...
#   export OPENAI_BASE_URL="https://gpt-agent.cc/v1"
#   nohup bash scripts/run_wikimia25_api_sampling_cache_queue.sh "0 1 2 3 4 5 6 7" 5 \
#     --params sampling_batch_size:100 \
#     > logs/simmia_cache_queue.wikimia25_api.nohup.log 2>&1 &
#
# Override API_WIKIMIA25_MODELS with the exact model ids returned by your
# OpenAI-compatible /models endpoint.

GPU_IDS="${1:-${GPU_IDS:-0 1 2 3 4 5 6 7}}"
CONCURRENCY="${2:-${CONCURRENCY:-5}}"
EXTRA_ARGS=("${@:3}")

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://gpt-agent.cc/v1}"
export METHODS="${METHODS:-hard soft}"
export RUN_WIKIMIA=0
export RUN_MIMIR=0
export RUN_WIKIMIA25=1
export RUN_API=1
export WIKIMIA25_MODELS=""
export WIKIMIA25_SUBSETS="${WIKIMIA25_SUBSETS:-paper_subset}"
export API_WIKIMIA25_MODELS="${API_WIKIMIA25_MODELS:-api:openai/claude-4-5-haiku api:openai/gemini-2.5-flash api:openai/gpt-5-chat-latest}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p logs

bash scripts/run_paper_sampling_cache_queue.sh \
  "$GPU_IDS" "$CONCURRENCY" \
  "${EXTRA_ARGS[@]}"
