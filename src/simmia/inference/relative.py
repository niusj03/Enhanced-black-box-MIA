import argparse
import numpy as np
from tqdm import tqdm

from typing import Tuple, List

from sentence_transformers.util import dot_score

from simmia._internal import INFERENCE_METHODS
from simmia.utils import get_start_idx


def _finite_scalar(value: float) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def _require_fields(record: dict, fields: List[str], inference_name: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise KeyError(
            f"{inference_name} requires records postprocessed with "
            "process_wpmia_word_data. Missing: " + ", ".join(missing)
        )


@INFERENCE_METHODS.register("relative_label_ratio")
def relative_label_ratio(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    for rec in tqdm(records, desc=f"Inferring with [{relative_label_ratio.__name__}]"):
        start_idx = get_start_idx(len(rec["label_results"]), args.prefix_ratio)
        orig = np.array(
            [
                x if x is not None else 10e-3
                for x in rec["label_freq_target"][start_idx:]
            ][: args.num_samples],
            dtype=float,
        )
        prep = np.array(
            [
                x if x is not None else 0.0
                for x in rec["prefix_label_freq_target"][start_idx:]
            ][: args.num_samples],
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


@INFERENCE_METHODS.register("wpmia_ll_score")
def wpmia_ll_score(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    required_fields = ["wpmia_L0"]

    for rec in tqdm(records, desc=f"Inferring with [{wpmia_ll_score.__name__}]"):
        _require_fields(rec, required_fields, wpmia_ll_score.__name__)
        predictions.append(_finite_scalar(rec["wpmia_L0"]))
        answers.append(rec["label"])

    return predictions, answers


@INFERENCE_METHODS.register("wpmia_nonmember_ratio_score")
def wpmia_nonmember_ratio_score(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    required_fields = ["wpmia_L0", "wpmia_Lnm"]

    for rec in tqdm(
        records, desc=f"Inferring with [{wpmia_nonmember_ratio_score.__name__}]"
    ):
        _require_fields(rec, required_fields, wpmia_nonmember_ratio_score.__name__)
        prediction = np.divide(rec["wpmia_Lnm"], rec["wpmia_L0"])
        predictions.append(_finite_scalar(prediction))
        answers.append(rec["label"])

    return predictions, answers


@INFERENCE_METHODS.register("wpmia_score")
def wpmia_score(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    required_fields = ["wpmia_L0", "wpmia_Lnm", "wpmia_Lm"]

    for rec in tqdm(records, desc=f"Inferring with [{wpmia_score.__name__}]"):
        _require_fields(rec, required_fields, wpmia_score.__name__)
        numerator = rec["wpmia_Lnm"] - args.wpmia_gamma * rec["wpmia_Lm"]
        prediction = np.divide(numerator, rec["wpmia_L0"])
        predictions.append(_finite_scalar(prediction))
        answers.append(rec["label"])

    return predictions, answers


@INFERENCE_METHODS.register("wpmia_word_nonmember_ratio_score")
def wpmia_word_nonmember_ratio_score(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    required_fields = ["wpmia_word_L0", "wpmia_word_Lnm"]

    for rec in tqdm(
        records,
        desc=f"Inferring with [{wpmia_word_nonmember_ratio_score.__name__}]",
    ):
        _require_fields(rec, required_fields, wpmia_word_nonmember_ratio_score.__name__)
        word_L0 = np.array(rec["wpmia_word_L0"], dtype=float)
        word_Lnm = np.array(rec["wpmia_word_Lnm"], dtype=float)
        prediction = np.sum(np.divide(word_Lnm, word_L0))
        predictions.append(_finite_scalar(prediction))
        answers.append(rec["label"])

    return predictions, answers


@INFERENCE_METHODS.register("wpmia_word_score")
def wpmia_word_score(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []
    required_fields = ["wpmia_word_L0", "wpmia_word_Lnm", "wpmia_word_Lm"]

    for rec in tqdm(records, desc=f"Inferring with [{wpmia_word_score.__name__}]"):
        _require_fields(rec, required_fields, wpmia_word_score.__name__)
        word_L0 = np.array(rec["wpmia_word_L0"], dtype=float)
        word_Lnm = np.array(rec["wpmia_word_Lnm"], dtype=float)
        word_Lm = np.array(rec["wpmia_word_Lm"], dtype=float)
        numerator = word_Lnm - args.wpmia_gamma * word_Lm
        prediction = np.sum(np.divide(numerator, word_L0))
        predictions.append(_finite_scalar(prediction))
        answers.append(rec["label"])

    return predictions, answers
