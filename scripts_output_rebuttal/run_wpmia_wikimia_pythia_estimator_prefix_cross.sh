#!/usr/bin/env bash
set -euo pipefail

# Rebuttal experiment: estimator x prefix crossed ablation for
# WikiMIA/Pythia-6.9B. This is a strictly offline recomputation over the cached
# first-word samples; it never invokes the target LLM or writes to simmia_out.
# The cosine estimator uses the fixed affine map (1 + cosine) / 2, while the
# semantic-kernel estimator uses exp((cosine - 1) / tau).
#
# Full run on GPU 0:
#   nohup bash \
#     scripts_output_rebuttal/run_wpmia_wikimia_pythia_estimator_prefix_cross.sh \
#     "0" \
#     > scripts_output_rebuttal/wikimia_pythia69b_estimator_prefix_cross.nohup.log 2>&1 &
#
# Cache-only validation:
#   DRY_RUN=1 LENGTHS="32" bash \
#     scripts_output_rebuttal/run_wpmia_wikimia_pythia_estimator_prefix_cross.sh "0"
#
# Recompute existing outputs:
#   OVERWRITE=1 bash \
#     scripts_output_rebuttal/run_wpmia_wikimia_pythia_estimator_prefix_cross.sh "0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS_RAW="${1:-${GPU_IDS:-0}}"
EXTRA_ARGS=("${@:2}")

MODEL="${MODEL:-EleutherAI/pythia-6.9b}"
LENGTHS="${LENGTHS:-32 64 128}"
NUM_SHOTS="${NUM_SHOTS:-7}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
WPMIA_TAU="${WPMIA_TAU:-0.2}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
EPSILON="${EPSILON:-1e-8}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-simmia_out/WikiMIA}"
OUTPUT_ROOT="${OUTPUT_ROOT:-scripts_output_rebuttal/estimator_prefix_cross}"
AUC_TABLE_PATH="${AUC_TABLE_PATH:-scripts_output_rebuttal/wikimia_pythia69b_estimator_prefix_cross_auc.csv}"
CHUNK_SIZE="${CHUNK_SIZE:-8}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-512}"
DEVICE="${DEVICE:-auto}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/logs/$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="${RUN_LOG:-${RUN_ROOT}/run.log}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD="$PYTHON_BIN"
elif [[ -x "$HOME/miniconda3/envs/simmia/bin/python" ]]; then
  PYTHON_CMD="$HOME/miniconda3/envs/simmia/bin/python"
else
  PYTHON_CMD=python
fi

normalized_gpu_ids="${GPU_IDS_RAW//,/ }"
read -r -a gpu_id_array <<< "$normalized_gpu_ids"
if [[ "$DEVICE" == cuda:* || "$DEVICE" == "auto" ]]; then
  if [[ ${#gpu_id_array[@]} -eq 0 ]]; then
    echo "No GPU ID was provided for DEVICE=$DEVICE" >&2
    exit 2
  fi
  export CUDA_VISIBLE_DEVICES="${gpu_id_array[0]}"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-simmia}}"
mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$MPLCONFIGDIR" "$(dirname "$AUC_TABLE_PATH")"

read -r -a length_array <<< "$LENGTHS"
cmd=(
  "$PYTHON_CMD"
  "$SCRIPT_DIR/compute_wikimia_pythia_estimator_prefix_cross.py"
  --source-cache-root "$SOURCE_CACHE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --auc-table-path "$AUC_TABLE_PATH"
  --model-name "$MODEL"
  --lengths "${length_array[@]}"
  --num-shots "$NUM_SHOTS"
  --num-samples "$NUM_SAMPLES"
  --tau "$WPMIA_TAU"
  --gamma "$WPMIA_GAMMA"
  --epsilon "$EPSILON"
  --embedding-model "$EMBEDDING_MODEL"
  --device "$DEVICE"
  --chunk-size "$CHUNK_SIZE"
  --encode-batch-size "$ENCODE_BATCH_SIZE"
)
if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi
cmd+=("${EXTRA_ARGS[@]}")

echo "Rebuttal estimator x prefix crossed ablation"
echo "Source cache: $SOURCE_CACHE_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Lengths: $LENGTHS"
echo "Model: $MODEL"
echo "Estimator config: tau=$WPMIA_TAU gamma=$WPMIA_GAMMA epsilon=$EPSILON"
echo "Embedding: $EMBEDDING_MODEL"
echo "Device: $DEVICE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Target-LLM sampling: disabled"

if [[ "$DRY_RUN" == "1" ]]; then
  cmd+=(--preflight-only)
  printf 'COMMAND:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  echo "Dry run passed. No result files were written."
  exit 0
fi

{
  printf 'COMMAND:'
  printf ' %q' "${cmd[@]}"
  printf '\n\n'
  "${cmd[@]}"
} 2>&1 | tee "$RUN_LOG"

echo "Finished estimator x prefix crossed ablation."
echo "AUC table: $AUC_TABLE_PATH"
echo "Run log: $RUN_LOG"
