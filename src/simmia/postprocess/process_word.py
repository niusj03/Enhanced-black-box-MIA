import argparse
import sentence_transformers
import math
import re
import numpy as np

from typing import Tuple, List, Any, Optional
from collections import Counter

from simmia._internal import POSTPROCESS_METHODS
from simmia.utils import extract_first_word, get_start_idx


__all__ = [
    "process_word_data",
    "process_relative_word_data",
    "process_relative_label_word_data",
    "process_wpmia_word_data",
]


_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(\D|$))")


def _normalize_commas(s: str) -> str:
    return _THOUSANDS_RE.sub("", s)


def is_number(s: str) -> bool:
    try:
        x = float(_normalize_commas(s).strip())
        return math.isfinite(x)
    except ValueError:
        return False


def parse_number(s: str) -> float:
    s = _normalize_commas(s)
    return float(s.strip())


def numbers_equal(
    a: str, b: str, *, abs_tol: float = 1e-6, rel_tol: float = 1e-9
) -> bool:
    return math.isclose(
        parse_number(a), parse_number(b), abs_tol=abs_tol, rel_tol=rel_tol
    )


def _process_word_data(
    label_results: List[str],
    sample_results: List[List[Tuple[str, int]]],
    embedding_model: sentence_transformers.SentenceTransformer,
    smoothing: bool,
    exact_match_number: bool,
    sample_count_limit: Optional[int] = None,
) -> Tuple[
    List[List[float]],
    List[List[float]],
    List[float],
]:
    sample_similarity = []
    sample_frequencies = []
    label_frequencies = []

    for label_token, sample_token_count in zip(label_results, sample_results):
        label_word = extract_first_word(label_token)
        sample_word_counter = _counter_first_words(
            sample_token_count, sample_count_limit=sample_count_limit
        )

        approx_label_word = None
        if label_word not in sample_word_counter:
            for word in sample_word_counter.keys():
                if word.lower().startswith(
                    label_word.lower()
                ) or label_word.lower().startswith(word.lower()):
                    approx_label_word = word
                    break

        # Laplace Smoothing
        if smoothing:
            for word in sample_word_counter.keys():
                sample_word_counter.update({word: 1})
            if approx_label_word is None:
                sample_word_counter.update({label_word: 1})

        # embed words & compute statistics
        embeddings = embedding_model.encode(
            [label_word] + list(sample_word_counter.keys()),
            show_progress_bar=False,
        )
        label_embedding = embeddings[0:1, :]
        sample_embeddings = embeddings[1:, :]

        sample_scores = embedding_model.similarity(
            label_embedding, sample_embeddings
        ).squeeze(0)

        # use exact math for numbers instead
        if exact_match_number and is_number(label_word):
            for i, w in enumerate(sample_word_counter.keys()):
                if is_number(w):
                    sample_scores[i] = 1.0 if numbers_equal(label_word, w) else -1.0

        sample_freqs = np.asarray(list(sample_word_counter.values())) / sum(
            sample_word_counter.values()
        )

        # save results
        sample_similarity.append(sample_scores.tolist())
        sample_frequencies.append(sample_freqs.tolist())
        label_frequencies.append(
            sample_word_counter[
                label_word if approx_label_word is None else approx_label_word
            ]
            / sum(sample_word_counter.values())
        )

    return sample_similarity, sample_frequencies, label_frequencies


def _process_label_word_data(
    label_results: List[str],
    sample_results: List[List[Tuple[str, int]]],
    smoothing: bool,
    sample_count_limit: Optional[int] = None,
) -> List[float]:
    label_frequencies = []

    for label_token, sample_token_count in zip(label_results, sample_results):
        label_word = extract_first_word(label_token)
        sample_word_counter = _counter_first_words(
            sample_token_count, sample_count_limit=sample_count_limit
        )

        approx_label_word = None
        if label_word not in sample_word_counter:
            for word in sample_word_counter.keys():
                if word.lower().startswith(
                    label_word.lower()
                ) or label_word.lower().startswith(word.lower()):
                    approx_label_word = word
                    break

        if smoothing:
            for word in list(sample_word_counter.keys()):
                sample_word_counter.update({word: 1})
            if approx_label_word is None:
                sample_word_counter.update({label_word: 1})

        label_key = label_word if approx_label_word is None else approx_label_word
        label_frequencies.append(
            sample_word_counter[label_key] / sum(sample_word_counter.values())
        )

    return label_frequencies


def _to_float_array(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def _sample_count_limit_from_args(args: argparse.Namespace) -> Optional[int]:
    sample_count_limit = getattr(args, "sample_count_limit", None)
    if sample_count_limit is None:
        return None
    sample_count_limit = int(sample_count_limit)
    if sample_count_limit <= 0:
        raise ValueError("--sample_count_limit must be > 0 when provided")
    return sample_count_limit


def _counter_first_words(
    sample_token_count: List[Tuple[str, int]],
    sample_count_limit: Optional[int] = None,
) -> Counter:
    sample_word_counter = Counter()
    remaining = sample_count_limit
    for token, count in sample_token_count:
        if remaining is not None:
            if remaining <= 0:
                break
            count = min(int(count), remaining)
            remaining -= count
        word = extract_first_word(token)
        sample_word_counter.update({word: count})
    return sample_word_counter


def _remember_word(word: str, words: List[str], seen: set) -> None:
    if word not in seen:
        seen.add(word)
        words.append(word)


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    embeddings = _to_float_array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return embeddings / norms


def _wpmia_record_log_likelihoods(
    label_results: List[str],
    sample_results_by_route: List[List[List[Tuple[str, int]]]],
    embedding_model: sentence_transformers.SentenceTransformer,
    tau: float,
    exact_match_number: bool,
    start_idx: int,
    sample_count_limit: Optional[int] = None,
    epsilon: float = 1e-8,
) -> Tuple[Tuple[float, float, float], Tuple[List[float], List[float], List[float]]]:
    prepared = []
    unique_words = []
    seen_words = set()

    for label_token, *route_samples in zip(
        label_results[start_idx:],
        *(route[start_idx:] for route in sample_results_by_route),
    ):
        label_word = extract_first_word(label_token)
        _remember_word(label_word, unique_words, seen_words)

        counters = [
            _counter_first_words(samples, sample_count_limit=sample_count_limit)
            for samples in route_samples
        ]
        for counter in counters:
            for word in counter.keys():
                _remember_word(word, unique_words, seen_words)
        prepared.append((label_word, counters))

    if not prepared:
        return (0.0, 0.0, 0.0), ([], [], [])

    embeddings = embedding_model.encode(unique_words, show_progress_bar=False)
    normalized_embeddings = _normalize_rows(embeddings)
    embedding_by_word = dict(zip(unique_words, normalized_embeddings))

    log_likelihoods = np.zeros(len(sample_results_by_route), dtype=float)
    word_log_likelihoods = [[] for _route in sample_results_by_route]
    for label_word, counters in prepared:
        label_embedding = embedding_by_word[label_word]
        label_is_number = exact_match_number and is_number(label_word)

        for route_idx, counter in enumerate(counters):
            total = sum(counter.values())
            if total <= 0:
                word_log_likelihood = float(np.log(epsilon))
                word_log_likelihoods[route_idx].append(word_log_likelihood)
                log_likelihoods[route_idx] += word_log_likelihood
                continue

            probability = 0.0
            for sample_word, count in counter.items():
                if label_is_number and is_number(sample_word):
                    score = (
                        1.0 if numbers_equal(label_word, sample_word) else -1.0
                    )
                else:
                    score = float(
                        np.dot(label_embedding, embedding_by_word[sample_word])
                    )
                probability += (count / total) * np.exp((score - 1.0) / tau)
            word_log_likelihood = float(np.log(epsilon + probability))
            word_log_likelihoods[route_idx].append(word_log_likelihood)
            log_likelihoods[route_idx] += word_log_likelihood

    return (
        tuple(float(x) for x in log_likelihoods),
        tuple(list(x) for x in word_log_likelihoods),
    )


@POSTPROCESS_METHODS.register("process_word_data")
def process_word_data(
    rank: int,
    args: argparse.Namespace,
    embedder: Any,
    instance: dict,
) -> dict:
    sample_similarity, sample_frequencies, label_frequencies = _process_word_data(
        label_results=instance["label_results"],
        sample_results=instance["sample_results"],
        embedding_model=embedder,
        smoothing=args.smoothing,
        exact_match_number=args.exact_match_number,
        sample_count_limit=_sample_count_limit_from_args(args),
    )

    instance.update(
        {
            "sem_target": sample_similarity,
            "freq_target": sample_frequencies,
            "label_freq_target": label_frequencies,
        }
    )
    return instance


@POSTPROCESS_METHODS.register("process_relative_word_data")
def process_relative_word_data(
    rank: int,
    args: argparse.Namespace,
    embedder: Any,
    instance: dict,
) -> dict:
    sample_similarity, sample_frequencies, label_frequencies = _process_word_data(
        label_results=instance["label_results"],
        sample_results=instance["sample_results"],
        embedding_model=embedder,
        smoothing=args.smoothing,
        exact_match_number=args.exact_match_number,
        sample_count_limit=_sample_count_limit_from_args(args),
    )
    prefix_sample_similarity, prefix_sample_frequencies, prefix_label_frequencies = (
        _process_word_data(
            label_results=instance["label_results"],
            sample_results=instance["nonmember_prefix_sample_results"],
            embedding_model=embedder,
            smoothing=args.smoothing,
            exact_match_number=args.exact_match_number,
            sample_count_limit=_sample_count_limit_from_args(args),
        )
    )

    instance.update(
        {
            "sem_target": sample_similarity,
            "freq_target": sample_frequencies,
            "label_freq_target": label_frequencies,
            "prefix_sem_target": prefix_sample_similarity,
            "prefix_freq_target": prefix_sample_frequencies,
            "prefix_label_freq_target": prefix_label_frequencies,
        }
    )
    return instance


@POSTPROCESS_METHODS.register("process_relative_label_word_data")
def process_relative_label_word_data(
    rank: int,
    args: argparse.Namespace,
    embedder: Any,
    instance: dict,
) -> dict:
    label_frequencies = _process_label_word_data(
        label_results=instance["label_results"],
        sample_results=instance["sample_results"],
        smoothing=args.smoothing,
        sample_count_limit=_sample_count_limit_from_args(args),
    )
    prefix_label_frequencies = _process_label_word_data(
        label_results=instance["label_results"],
        sample_results=instance["nonmember_prefix_sample_results"],
        smoothing=args.smoothing,
        sample_count_limit=_sample_count_limit_from_args(args),
    )

    instance.update(
        {
            "label_freq_target": label_frequencies,
            "prefix_label_freq_target": prefix_label_frequencies,
        }
    )
    return instance


@POSTPROCESS_METHODS.register("process_wpmia_word_data")
def process_wpmia_word_data(
    rank: int,
    args: argparse.Namespace,
    embedder: Any,
    instance: dict,
) -> dict:
    required_fields = [
        "label_results",
        "sample_results",
        "nonmember_prefix_sample_results",
        "member_prefix_sample_results",
    ]
    missing = [field for field in required_fields if field not in instance]
    if missing:
        raise KeyError(
            "WPMIA requires new-format records.jsonl with explicit raw, "
            "nonmember-prefix, and member-prefix sampling fields. Missing: "
            + ", ".join(missing)
        )

    start_idx = get_start_idx(len(instance["label_results"]), args.prefix_ratio)
    (wpmia_L0, wpmia_Lnm, wpmia_Lm), (
        wpmia_word_L0,
        wpmia_word_Lnm,
        wpmia_word_Lm,
    ) = _wpmia_record_log_likelihoods(
        label_results=instance["label_results"],
        sample_results_by_route=[
            instance["sample_results"],
            instance["nonmember_prefix_sample_results"],
            instance["member_prefix_sample_results"],
        ],
        embedding_model=embedder,
        tau=args.wpmia_tau,
        exact_match_number=args.exact_match_number,
        start_idx=start_idx,
        sample_count_limit=_sample_count_limit_from_args(args),
    )

    instance.update(
        {
            "wpmia_L0": wpmia_L0,
            "wpmia_Lnm": wpmia_Lnm,
            "wpmia_Lm": wpmia_Lm,
            "wpmia_word_L0": wpmia_word_L0,
            "wpmia_word_Lnm": wpmia_word_Lnm,
            "wpmia_word_Lm": wpmia_word_Lm,
        }
    )
    return instance
