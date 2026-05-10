import argparse
import sentence_transformers
import math
import re
import numpy as np

from typing import Tuple, List, Any
from collections import Counter

from simmia._internal import POSTPROCESS_METHODS
from simmia.utils import extract_first_word


__all__ = [
    "process_word_data",
    "process_relative_word_data",
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
        sample_word_counter = Counter()
        for token, count in sample_token_count:
            word = extract_first_word(token)
            sample_word_counter.update({word: count})

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
    )
    prefix_sample_similarity, prefix_sample_frequencies, prefix_label_frequencies = (
        _process_word_data(
            label_results=instance["label_results"],
            sample_results=instance["prefix_sample_results"],
            embedding_model=embedder,
            smoothing=args.smoothing,
            exact_match_number=args.exact_match_number,
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
