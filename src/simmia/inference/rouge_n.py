import argparse
import re
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import Tuple, List

from simmia._internal import INFERENCE_METHODS


def get_suffix(text: str, prefix_length: int) -> list:
    """
    Extracts a suffix from the given text, based on the specified prefix ratio and text length.
    """
    words = text.split(" ")
    words = [word for word in words if word != ""]
    words = words[prefix_length:]
    return words


def ngrams(sequence, n) -> zip:
    """
    Generates n-grams from a sequence.
    """
    return zip(*[sequence[i:] for i in range(n)])


def rouge_n(candidate: list, reference: list, n=1) -> float:
    """
    Calculates the ROUGE-N score between a candidate and a reference.
    """
    if not candidate or not reference:
        return 0
    candidate_ngrams = list(ngrams(candidate, n))
    reference_ngrams = list(ngrams(reference, n))
    ref_words_count = Counter(reference_ngrams)
    cand_words_count = Counter(candidate_ngrams)
    overlap = ref_words_count & cand_words_count
    recall = sum(overlap.values()) / len(reference)
    precision = sum(overlap.values()) / len(candidate)
    return recall


def clean_text(text: str, model_name: str) -> str:
    """
    Removes specific special tokens from the text based on the model's output.
    """
    if "pythia" in model_name or "gpt" in model_name or "qwen" in model_name:
        return re.sub(r"<\|endoftext\|>", "", text)
    elif "llama" in model_name or "opt" in model_name:
        text = re.sub(r"<s> ", "", text)
        return re.sub(r"</s>", "", text)
    return text


@INFERENCE_METHODS.register("rouge_n")
def rouge_n_inference(
    args: argparse.Namespace, records: List[dict]
) -> Tuple[List[float], List[bool]]:
    predictions = []
    answers = []

    for rec in tqdm(records, desc="Inferring with [rouge_1]"):
        suffix_ref = rec["suffix"].split()
        prefix = rec["prefix"]
        rouge_scores = []
        prefix_words = [word for word in prefix.split(" ") if word != ""]
        prefix_length = len(prefix_words)

        sample_texts = rec["sample_results"]

        for cand in sample_texts:
            text_output = clean_text(cand, args.model_name_or_path)
            suffix_cand = get_suffix(text_output, prefix_length)
            rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1))

        if rouge_scores:
            y_pred = float(np.mean(rouge_scores))
        else:
            y_pred = 0.0

        predictions.append(y_pred)
        answers.append(rec["label"])

    assert len(predictions) == len(answers) and len(answers) > 0

    return predictions, answers
