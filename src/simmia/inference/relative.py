import argparse
import numpy as np
from tqdm import tqdm

from typing import Tuple, List

from sentence_transformers.util import dot_score

from simmia._internal import INFERENCE_METHODS
from simmia.utils import get_start_idx


@INFERENCE_METHODS.register("relative_label_ratio")
def relative_label_ratio(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    for rec in tqdm(records, desc=f"Inferring with [{relative_label_ratio.__name__}]"):
        orig = np.array(
            [x if x is not None else 10e-3 for x in rec["label_freq_target"]][
                : args.num_samples
            ],
            dtype=float,
        )
        prep = np.array(
            [x if x is not None else 0.0 for x in rec["prefix_label_freq_target"]][
                : args.num_samples
            ],
            dtype=float,
        )
        ratio = np.nan_to_num(prep / (orig))
        y_pred = -float(np.mean(ratio))
        predictions.append(y_pred)
        answers.append(rec["label"])
    return predictions, answers


@INFERENCE_METHODS.register("relative_semantic_ratio")
def relative_semantic_ratio(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []

    for rec in tqdm(
        records, desc=f"Inferring with [{relative_semantic_ratio.__name__}]"
    ):
        start_idx = get_start_idx(len(rec["label_results"]), args.prefix_ratio)
        freq_target = rec["freq_target"]
        pre_freq_target = rec["prefix_freq_target"]
        freq_target = freq_target[start_idx:]
        pre_freq_target = pre_freq_target[start_idx:]

        sem_target = [(np.array(p) + 1) / 2 for p in rec["sem_target"][start_idx:]]
        pre_sem_target = [
            (np.array(p) + 1) / 2 for p in rec["prefix_sem_target"][start_idx:]
        ]
        orig = [
            float(dot_score(np.array(p), np.array(q)))
            for p, q in zip(sem_target, freq_target)
        ]
        prep = [
            float(dot_score(np.array(p), np.array(q)))
            for p, q in zip(pre_sem_target, pre_freq_target)
        ]
        orig_arr = np.array(orig, dtype=float)
        prep_arr = np.array(prep, dtype=float)

        ratio = np.nan_to_num(prep_arr / (orig_arr + 10e-12))

        y_pred = -float(np.mean(ratio))
        predictions.append(y_pred)
        answers.append(rec["label"])

    return predictions, answers
