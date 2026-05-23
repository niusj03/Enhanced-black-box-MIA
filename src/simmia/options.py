import argparse

from simmia._internal import (
    SAMPLING_METHODS,
    ASYNC_SAMPLING_METHODS,
    POSTPROCESS_METHODS,
    INFERENCE_METHODS,
)


def _positive_float(value: str) -> float:
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return value


def _nonnegative_float(value: str) -> float:
    value = float(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def add_arguments(parser: argparse.ArgumentParser):
    # Basic
    parser.add_argument(
        "--gpu_ids", type=str, nargs="+", default=[], help="the GPUs to use"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="the max concurrency for calling APIs",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="EleutherAI/pythia-6.9b",
        help="the model to attack",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the directory to store the result",
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default=None,
        help="optional prefix for the ROC plot filename",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="whether to rewrite the cached result"
    )

    # Data
    parser.add_argument(
        "--data",
        type=str,
        choices=["swj0419/WikiMIA", "iamgroot42/mimir", "SimMIA/WikiMIA-25"],
        default="swj0419/WikiMIA",
        help="the dataset to evaluate",
    )
    parser.add_argument(
        "--sub_dataset",
        type=str,
        default=None,
        help="the subset of the dataset to evaluate",
    )
    split_choices = ["ngram_7_0.2", "ngram_13_0.2"]
    parser.add_argument(
        "--split",
        type=str,
        choices=split_choices,
        default="ngram_7_0.2",
        help=f"Required if `--data iamgroot42/mimir`. Choices: {split_choices}",
    )

    # Preprocess
    parser.add_argument("--num_shots", type=int, default=1, help="the prefix shots")
    parser.add_argument(
        "--prefix_len", type=int, default=-1, help="the length of the non member prfix."
    )
    parser.add_argument(
        "--prefix_ratio",
        type=float,
        default=0.0,
        help="the part that will not calculate as test",
    )

    # Sampling
    parser.add_argument(
        "--sampling",
        type=str,
        default="relative_word_by_word",
        choices=set(SAMPLING_METHODS.list_keys() + ASYNC_SAMPLING_METHODS.list_keys()),
        help="the function name to sample for each test instance",
    )
    parser.add_argument(
        "--top_k", type=int, default=20, help="the top_k value of sampling generation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="the number of samples of sampling generation",
    )
    parser.add_argument("--seed", type=int, default=42, help="the random seed")

    # Postprocess
    parser.add_argument(
        "--postprocess",
        type=str,
        default=None,
        choices=POSTPROCESS_METHODS.list_keys(),
        help="the function name to postprocess each test instance",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="the model used to compute semantic similarity",
    )
    parser.add_argument(
        "--smoothing", action="store_true", help="whether to apply Laplace Smoothing"
    )
    parser.add_argument(
        "--exact_match_number",
        action="store_true",
        help="compute exact match for numbers",
    )

    # Membership inference
    parser.add_argument(
        "--inference",
        type=str,
        default="relative_semantic_ratio",
        choices=INFERENCE_METHODS.list_keys(),
        help="the function name to infer the membership for each test instance",
    )
    parser.add_argument(
        "--wpmia_tau",
        type=_positive_float,
        default=0.1,
        help="temperature for WPMIA semantic-kernel likelihood scoring",
    )
    parser.add_argument(
        "--wpmia_gamma",
        type=_nonnegative_float,
        default=1.0,
        help="member-prefix contrast weight for WPMIA scoring",
    )

    # Additional
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="the additional hyperparameter setting, e.g., `decoding:greedy`",
    )

    return parser
