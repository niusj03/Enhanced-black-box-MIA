#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?model_name_or_path}"
DATA="${2:?data}"
SUB_DATASET="${3:?sub_dataset}"
shift 3

GPU_IDS="${1-}";       shift $(( $# > 0 ? 1 : 0 ))
CONCURRENCY="${1-}";   shift $(( $# > 0 ? 1 : 0 ))

DATA_LC="${DATA,,}"
SUB_DATASET_LC="${SUB_DATASET,,}"
WPMIA_TAU="${WPMIA_TAU:-0.1}"
WPMIA_GAMMA="${WPMIA_GAMMA:-1.0}"
RESULT_NAME="${WPMIA_RESULT_NAME:-wpmia_tau_${WPMIA_TAU}_gamma_${WPMIA_GAMMA}}"

if [[ "$DATA_LC" == *wikimia* && "$SUB_DATASET_LC" =~ ^[0-9]+$ ]]; then
  SUB_DATASET="WikiMIA_length${SUB_DATASET}"
  SUB_DATASET_LC="${SUB_DATASET,,}"
fi

in_list() {
  local needle="$1"; shift
  local item
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

validate_subdataset() {
  local data="$1" sub="$2"; shift 2
  local -a allowed=( "$@" )

  (( ${#allowed[@]} == 0 )) && return 0

  if ! in_list "$sub" "${allowed[@]}"; then
    {
      echo "ERROR: invalid sub_dataset '$sub' for data '$data'"
      echo "Allowed:"
      printf '  - %s\n' "${allowed[@]}"
    } >&2
    exit 2
  fi
}

allowed_subdatasets=()
case "$DATA_LC" in
  *mimir*)
    allowed_subdatasets=(
      "wikipedia_(en)"
      "github"
      "pile_cc"
      "pubmed_central"
      "arxiv"
      "dm_mathematics"
      "hackernews"
    )
    ;;
  *wikimia-25*)
    allowed_subdatasets=(
      "WikiMIA_length32"
      "WikiMIA_length64"
      "WikiMIA_length128"
      "paper_subset"
    )
    ;;
  *wikimia*)
    allowed_subdatasets=(
      "WikiMIA_length32"
      "WikiMIA_length64"
      "WikiMIA_length128"
      "WikiMIA_length256"
    )
    ;;
esac

validate_subdataset "$DATA" "$SUB_DATASET" "${allowed_subdatasets[@]}"
SUB_DATASET_LC="${SUB_DATASET,,}"

if [[ "$DATA_LC" == *mimir* ]]; then
  NUM_SHOTS=10
else
  NUM_SHOTS=7
fi

if [[ "$DATA_LC" == *-25* ]]; then
  NUM_SAMPLES=10
else
  NUM_SAMPLES=100
fi

cmd=(
  simmia.benchmark
  --model_name_or_path "$MODEL"
  --sampling relative_word_by_word
  --postprocess process_wpmia_word_data
  --inference wpmia_score
  --output_dir simmia_out
  --result_name "$RESULT_NAME"
  --num_samples "$NUM_SAMPLES"
  --data "$DATA"
  --sub_dataset "$SUB_DATASET"
  --num_shots "$NUM_SHOTS"
  --prefix_ratio 0.0
  --top_k 20
  --wpmia_tau "$WPMIA_TAU"
  --wpmia_gamma "$WPMIA_GAMMA"
)

if [[ "$SUB_DATASET_LC" == "dm_mathematics" ]]; then
  cmd+=(--exact_match_number)
fi
if [[ -n "${GPU_IDS:-}" ]]; then
  GPU_IDS="${GPU_IDS//,/ }"
  read -r -a GPU_ID_ARR <<< "$GPU_IDS"
  cmd+=(--gpu_ids "${GPU_ID_ARR[@]}")
fi

[[ -n "${CONCURRENCY:-}" ]] && cmd+=(--concurrency "$CONCURRENCY")

cmd+=("$@")
"${cmd[@]}"
