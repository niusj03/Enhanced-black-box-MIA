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

if [[ "$DATA_LC" == *wikimia* && "$SUB_DATASET_LC" =~ ^[0-9]+$ ]]; then
  SUB_DATASET="WikiMIA_length${SUB_DATASET}"
  SUB_DATASET_LC="${SUB_DATASET,,}"   # keep LC version in sync (optional but nice)
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

  # If no rules apply, skip validation
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

# Decide allowed sub_datasets by DATA
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

if [[ "$DATA_LC" == *mimir* ]]; then
  NUM_SHOTS=10
  MAX_LENGTH=2048
else
  NUM_SHOTS=7
  MAX_LENGTH=1024
fi

cmd=(
  simmia.benchmark
  --model_name_or_path "$MODEL"
  --sampling generate_all_remaining
  --inference rouge_n
  --output_dir samia_out
  --num_samples 10
  --data "$DATA"
  --sub_dataset "$SUB_DATASET"
  --num_shots "$NUM_SHOTS"
  --prefix_ratio 0.5
  --top_k 50
  --params "max_length:$MAX_LENGTH"
)

if [[ -n "${GPU_IDS:-}" ]]; then
  # Accept either "0,1,2" or "0 1 2"
  GPU_IDS="${GPU_IDS//,/ }"
  read -r -a GPU_ID_ARR <<< "$GPU_IDS"
  cmd+=(--gpu_ids "${GPU_ID_ARR[@]}")
fi

[[ -n "${CONCURRENCY:-}" ]] && cmd+=(--concurrency "$CONCURRENCY")

cmd+=("$@")
"${cmd[@]}"
