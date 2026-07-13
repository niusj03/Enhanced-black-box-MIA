#!/usr/bin/env python3
"""Offline estimator x prefix ablation for WikiMIA/Pythia-6.9B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from simmia.utils import extract_first_word  # noqa: E402


ROUTES = (
    ("raw", "sample_results"),
    ("nonmember", "nonmember_prefix_sample_results"),
    ("member", "member_prefix_sample_results"),
)
ESTIMATORS = ("exact_match", "cosine_similarity", "semantic_kernel")
PREFIX_MODES = (
    "no_prefix",
    "nonmember_prefix",
    "member_nonmember_contrast",
)


@dataclass
class PreparedRecord:
    index: int
    label: int
    positions: list[tuple[str, tuple[Counter[str], Counter[str], Counter[str]]]]


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a finite number > 0")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite number >= 0")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact-match, linearly normalized cosine, and semantic-kernel WPMIA "
            "estimators under three prefix constructions from cached records."
        )
    )
    parser.add_argument(
        "--source-cache-root", default="simmia_out/WikiMIA", type=Path
    )
    parser.add_argument(
        "--output-root",
        default="scripts_output_rebuttal/estimator_prefix_cross",
        type=Path,
    )
    parser.add_argument(
        "--auc-table-path",
        default="scripts_output_rebuttal/wikimia_pythia69b_estimator_prefix_cross_auc.csv",
        type=Path,
    )
    parser.add_argument(
        "--model-name", default="EleutherAI/pythia-6.9b"
    )
    parser.add_argument("--lengths", nargs="+", type=positive_int, default=[32, 64, 128])
    parser.add_argument("--num-shots", type=positive_int, default=7)
    parser.add_argument("--num-samples", type=positive_int, default=100)
    parser.add_argument("--tau", type=positive_float, default=0.2)
    parser.add_argument("--gamma", type=nonnegative_float, default=1.0)
    parser.add_argument("--epsilon", type=positive_float, default=1e-8)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SentenceTransformer device, for example cuda:0, cpu, or auto.",
    )
    parser.add_argument("--chunk-size", type=positive_int, default=8)
    parser.add_argument("--encode-batch-size", type=positive_int, default=512)
    parser.add_argument(
        "--max-records",
        type=positive_int,
        default=None,
        help="Optional per-length record cap for smoke tests only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def records_path(args: argparse.Namespace, length: int) -> Path:
    return (
        args.source_cache_root
        / f"WikiMIA_length{length}"
        / args.model_name.rsplit("/", 1)[-1]
        / str(args.num_shots)
        / "records.jsonl"
    )


def first_word_counter(samples: Sequence[Sequence[object]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for sample in samples:
        if len(sample) != 2:
            raise ValueError(f"invalid cached sample entry: {sample!r}")
        token, raw_count = sample
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"sample count must be nonnegative: {sample!r}")
        counter.update({extract_first_word(str(token)): count})
    return counter


def prepare_record(
    record: dict,
    *,
    length: int,
    index: int,
    expected_samples: int,
) -> PreparedRecord:
    required = ["label", "label_results", *(field for _, field in ROUTES)]
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f"length={length} record={index} is missing fields: {', '.join(missing)}"
        )

    label_results = record["label_results"]
    route_results = [record[field] for _, field in ROUTES]
    route_lengths = [len(route) for route in route_results]
    if any(route_length != len(label_results) for route_length in route_lengths):
        raise ValueError(
            f"length={length} record={index} has unaligned routes: "
            f"labels={len(label_results)}, routes={route_lengths}"
        )

    positions = []
    for word_index, values in enumerate(zip(label_results, *route_results)):
        label_word = extract_first_word(str(values[0]))
        counters = tuple(first_word_counter(samples) for samples in values[1:])
        for route_index, counter in enumerate(counters):
            total = sum(counter.values())
            if total != expected_samples:
                route_name = ROUTES[route_index][0]
                raise ValueError(
                    f"length={length} record={index} word={word_index} "
                    f"route={route_name} has {total} samples; "
                    f"expected exactly {expected_samples}"
                )
        positions.append((label_word, counters))

    label = int(record["label"])
    if label not in (0, 1):
        raise ValueError(
            f"length={length} record={index} has non-binary label {label!r}"
        )
    return PreparedRecord(index=index, label=label, positions=positions)


def iter_record_chunks(
    path: Path,
    *,
    length: int,
    expected_samples: int,
    chunk_size: int,
    max_records: int | None,
) -> Iterator[list[PreparedRecord]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing records cache: {path}")

    chunk: list[PreparedRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_records is not None and index >= max_records:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"length={length} record={index} contains invalid JSON"
                ) from exc
            chunk.append(
                prepare_record(
                    record,
                    length=length,
                    index=index,
                    expected_samples=expected_samples,
                )
            )
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def collect_words(records: Iterable[PreparedRecord]) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    for record in records:
        for label_word, counters in record.positions:
            if label_word not in seen:
                seen.add(label_word)
                words.append(label_word)
            for counter in counters:
                for sample_word in counter:
                    if sample_word not in seen:
                        seen.add(sample_word)
                        words.append(sample_word)
    return words


def normalized_embedding_map(
    model: object,
    words: Sequence[str],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    embeddings = np.asarray(
        model.encode(
            list(words),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ),
        dtype=float,
    )
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1.0, norms)
    return dict(zip(words, embeddings))


def position_masses(
    label_word: str,
    counters: tuple[Counter[str], Counter[str], Counter[str]],
    embeddings: dict[str, np.ndarray],
    *,
    tau: float,
) -> dict[str, tuple[float, float, float]]:
    masses = {estimator: [] for estimator in ESTIMATORS}
    label_embedding = embeddings[label_word]

    for counter in counters:
        total = float(sum(counter.values()))
        exact_mass = 0.0
        cosine_similarity_mass = 0.0
        semantic_mass = 0.0
        for sample_word, count in counter.items():
            weight = count / total
            cosine = float(np.dot(label_embedding, embeddings[sample_word]))
            exact_mass += weight * float(sample_word == label_word)
            cosine_similarity_mass += weight * ((1.0 + cosine) / 2.0)
            semantic_mass += weight * math.exp((cosine - 1.0) / tau)
        masses["exact_match"].append(exact_mass)
        masses["cosine_similarity"].append(cosine_similarity_mass)
        masses["semantic_kernel"].append(semantic_mass)

    return {
        estimator: tuple(float(value) for value in values)
        for estimator, values in masses.items()
    }


def record_log_likelihoods(
    record: PreparedRecord,
    embeddings: dict[str, np.ndarray],
    *,
    length: int,
    tau: float,
    epsilon: float,
) -> dict[str, tuple[float, float, float]]:
    likelihoods = {estimator: np.zeros(3, dtype=float) for estimator in ESTIMATORS}

    for word_index, (label_word, counters) in enumerate(record.positions):
        masses = position_masses(label_word, counters, embeddings, tau=tau)
        for estimator, route_masses in masses.items():
            for route_index, mass in enumerate(route_masses):
                log_argument = epsilon + mass
                if log_argument <= 0 or not math.isfinite(log_argument):
                    route_name = ROUTES[route_index][0]
                    raise ValueError(
                        f"invalid {estimator} log argument: length={length} "
                        f"record={record.index} word={word_index} route={route_name} "
                        f"mass={mass:.17g} epsilon={epsilon:.17g}"
                    )
                likelihoods[estimator][route_index] += math.log(log_argument)

    return {
        estimator: tuple(float(value) for value in values)
        for estimator, values in likelihoods.items()
    }


def finite_ratio(numerator: float, denominator: float) -> float:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        value = np.divide(numerator, denominator)
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def membership_scores(
    likelihoods: tuple[float, float, float], gamma: float
) -> dict[str, float]:
    L0, Lnm, Lm = likelihoods
    return {
        "no_prefix": float(L0),
        "nonmember_prefix": finite_ratio(Lnm, L0),
        "member_nonmember_contrast": finite_ratio(Lnm - gamma * Lm, L0),
    }


def ensure_outputs_available(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n  ".join(str(path) for path in existing)
        raise FileExistsError(
            "refusing to overwrite existing outputs; pass --overwrite:\n  " + joined
        )


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_preflight(args: argparse.Namespace) -> None:
    for length in args.lengths:
        path = records_path(args, length)
        count = 0
        for chunk in iter_record_chunks(
            path,
            length=length,
            expected_samples=args.num_samples,
            chunk_size=args.chunk_size,
            max_records=args.max_records,
        ):
            count += len(chunk)
        if count == 0:
            raise ValueError(f"records cache is empty: {path}")
        print(f"[PREFLIGHT] length={length} records={count} cache={path}")


def run_self_test() -> None:
    epsilon = 1e-8
    tau = 0.2
    embeddings = {
        "target": np.asarray([1.0, 0.0]),
        "other": np.asarray([0.0, 1.0]),
    }
    counters = (
        Counter({"target": 1, "other": 1}),
        Counter({"target": 2}),
        Counter({"other": 2}),
    )
    masses = position_masses("target", counters, embeddings, tau=tau)
    expected_semantic_mid = 0.5 * (1.0 + math.exp(-1.0 / tau))
    assert np.allclose(masses["exact_match"], (0.5, 1.0, 0.0))
    assert np.allclose(masses["cosine_similarity"], (0.75, 1.0, 0.5))
    assert np.allclose(
        masses["semantic_kernel"],
        (expected_semantic_mid, 1.0, math.exp(-1.0 / tau)),
    )

    likelihoods = tuple(math.log(epsilon + value) for value in masses["exact_match"])
    scores = membership_scores(likelihoods, gamma=1.0)
    assert np.isclose(scores["no_prefix"], likelihoods[0])
    assert np.isclose(scores["nonmember_prefix"], likelihoods[1] / likelihoods[0])
    assert np.isclose(
        scores["member_nonmember_contrast"],
        (likelihoods[1] - likelihoods[2]) / likelihoods[0],
    )
    print("Synthetic estimator and sequence-score self-test passed.")


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def run_experiment(args: argparse.Namespace) -> None:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score

    detail_path = (
        args.output_root
        / "wikimia_pythia69b_estimator_prefix_cross_details.csv"
    )
    score_path = (
        args.output_root
        / "wikimia_pythia69b_estimator_prefix_cross_scores.csv"
    )
    output_paths = [args.auc_table_path, detail_path, score_path]
    ensure_outputs_available(output_paths, args.overwrite)

    device = resolve_device(args.device)
    print(f"Loading embedding model {args.embedding_model} on {device}")
    model = SentenceTransformer(args.embedding_model, device=device)
    model.eval()

    score_rows: list[dict] = []
    scores_by_setting: dict[tuple[int, str, str], tuple[list[int], list[float]]] = {}
    source_paths: dict[int, Path] = {}

    for length in args.lengths:
        source_path = records_path(args, length)
        source_paths[length] = source_path
        processed = 0
        for chunk in iter_record_chunks(
            source_path,
            length=length,
            expected_samples=args.num_samples,
            chunk_size=args.chunk_size,
            max_records=args.max_records,
        ):
            words = collect_words(chunk)
            embeddings = normalized_embedding_map(
                model, words, batch_size=args.encode_batch_size
            )
            for record in chunk:
                record_likelihoods = record_log_likelihoods(
                    record,
                    embeddings,
                    length=length,
                    tau=args.tau,
                    epsilon=args.epsilon,
                )
                for estimator in ESTIMATORS:
                    L0, Lnm, Lm = record_likelihoods[estimator]
                    setting_scores = membership_scores(
                        record_likelihoods[estimator], args.gamma
                    )
                    for prefix_mode in PREFIX_MODES:
                        score = setting_scores[prefix_mode]
                        key = (length, estimator, prefix_mode)
                        labels, scores = scores_by_setting.setdefault(key, ([], []))
                        labels.append(record.label)
                        scores.append(score)
                        score_rows.append(
                            {
                                "length": length,
                                "index": record.index,
                                "label": record.label,
                                "estimator": estimator,
                                "prefix_mode": prefix_mode,
                                "L0": f"{L0:.17g}",
                                "Lnm": f"{Lnm:.17g}",
                                "Lm": f"{Lm:.17g}",
                                "score": f"{score:.17g}",
                            }
                        )
            processed += len(chunk)
            print(
                f"[PROGRESS] length={length} records={processed}",
                flush=True,
            )
        if processed == 0:
            raise ValueError(f"records cache is empty: {source_path}")

    detail_rows = []
    auc_by_setting: dict[tuple[str, str, int], float] = {}
    for estimator in ESTIMATORS:
        for prefix_mode in PREFIX_MODES:
            for length in args.lengths:
                labels, scores = scores_by_setting[(length, estimator, prefix_mode)]
                if len(set(labels)) != 2:
                    raise ValueError(
                        f"length={length} has no binary class coverage for AUC"
                    )
                auc_pct = float(roc_auc_score(labels, scores) * 100.0)
                auc_by_setting[(estimator, prefix_mode, length)] = auc_pct
                member_scores = [
                    score for label, score in zip(labels, scores) if label == 1
                ]
                nonmember_scores = [
                    score for label, score in zip(labels, scores) if label == 0
                ]
                detail_rows.append(
                    {
                        "estimator": estimator,
                        "prefix_mode": prefix_mode,
                        "length": length,
                        "auc_pct": f"{auc_pct:.6f}",
                        "num_examples": len(labels),
                        "member_mean_score": f"{np.mean(member_scores):.17g}",
                        "nonmember_mean_score": f"{np.mean(nonmember_scores):.17g}",
                        "min_score": f"{min(scores):.17g}",
                        "max_score": f"{max(scores):.17g}",
                        "model": args.model_name,
                        "num_shots": args.num_shots,
                        "num_samples": args.num_samples,
                        "tau": args.tau,
                        "gamma": args.gamma,
                        "epsilon": args.epsilon,
                        "embedding_model": args.embedding_model,
                        "source_records": str(source_paths[length]),
                        "score_path": str(score_path),
                    }
                )

    wide_rows = []
    for estimator in ESTIMATORS:
        for prefix_mode in PREFIX_MODES:
            row = {"estimator": estimator, "prefix_mode": prefix_mode}
            for length in args.lengths:
                row[f"len{length}_auc_pct"] = (
                    f"{auc_by_setting[(estimator, prefix_mode, length)]:.2f}"
                )
            wide_rows.append(row)

    atomic_write_csv(
        score_path,
        [
            "length",
            "index",
            "label",
            "estimator",
            "prefix_mode",
            "L0",
            "Lnm",
            "Lm",
            "score",
        ],
        score_rows,
    )
    atomic_write_csv(
        detail_path,
        [
            "estimator",
            "prefix_mode",
            "length",
            "auc_pct",
            "num_examples",
            "member_mean_score",
            "nonmember_mean_score",
            "min_score",
            "max_score",
            "model",
            "num_shots",
            "num_samples",
            "tau",
            "gamma",
            "epsilon",
            "embedding_model",
            "source_records",
            "score_path",
        ],
        detail_rows,
    )
    atomic_write_csv(
        args.auc_table_path,
        ["estimator", "prefix_mode"]
        + [f"len{length}_auc_pct" for length in args.lengths],
        wide_rows,
    )

    print("\nEstimator x prefix AUC (%):")
    for row in wide_rows:
        values = " ".join(
            f"len{length}={row[f'len{length}_auc_pct']}"
            for length in args.lengths
        )
        print(f"  {row['estimator']:15s} {row['prefix_mode']:28s} {values}")
    print(f"AUC table: {args.auc_table_path}")
    print(f"Details:   {detail_path}")
    print(f"Scores:    {score_path}")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.preflight_only:
        run_preflight(args)
        return
    run_experiment(args)


if __name__ == "__main__":
    main()
