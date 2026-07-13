#!/usr/bin/env python3
"""Compute the 7-shot WPMIA prefix-mechanism diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np


os.environ.setdefault(
    "MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'simmia')}"
)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


REQUIRED_FIELDS = {
    "input",
    "label",
    "label_results",
    "sample_results",
    "nonmember_prefix_sample_results",
    "member_prefix_sample_results",
}

OUTPUT_FIELDS = [
    "shot",
    "target_id",
    "target_label",
    "L0",
    "Lm",
    "Lnm",
    "delta_m",
    "delta_nm",
    "tau",
    "embedding_model",
    "prefix_ratio",
    "sample_count_limit",
    "cache_path",
]


def jsonl_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def nonempty_line_count(path: Path) -> int:
    with path.open("rb") as fin:
        return sum(1 for line in fin if line.strip())


def target_id(text: str, label: int) -> str:
    payload = f"{int(label)}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_target_map(path: Path) -> dict[str, int]:
    targets: dict[str, int] = {}
    for row in jsonl_rows(path):
        label = int(row["label"])
        identifier = target_id(row["input"], label)
        if identifier in targets:
            raise ValueError(f"Duplicate target in {path}: {identifier}")
        targets[identifier] = label
    if not targets:
        raise ValueError(f"Empty full dataset: {path}")
    return targets


def preflight(args: argparse.Namespace) -> tuple[Path, dict[str, int]]:
    directory = args.cache_root / str(args.shot)
    records_path = directory / "records.jsonl"
    full_path = directory / "full_dataset.jsonl"
    prefix_path = directory / "prefix_data.json"
    for path in (records_path, full_path, prefix_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing cache file: {path}")

    targets = load_target_map(full_path)
    record_count = nonempty_line_count(records_path)
    if record_count != len(targets):
        raise ValueError(
            f"Incomplete cache: {record_count} records for {len(targets)} targets"
        )

    first_record = next(jsonl_rows(records_path))
    missing = REQUIRED_FIELDS - set(first_record)
    if missing:
        raise ValueError(f"Cache is missing WPMIA fields: {sorted(missing)}")

    with prefix_path.open("r", encoding="utf-8") as fin:
        prefix_data = json.load(fin)
    for prefix_type in ("member", "nonmember"):
        if len(prefix_data.get(prefix_type, [])) < args.shot:
            raise ValueError(
                f"{prefix_path} has fewer than {args.shot} {prefix_type} prefixes"
            )

    if args.max_targets is not None:
        if args.max_targets < 2:
            raise ValueError("--max_targets must be at least 2")
        per_label = max(1, args.max_targets // 2)
        selected = []
        for label in (0, 1):
            selected.extend(
                sorted(identifier for identifier, value in targets.items() if value == label)[
                    :per_label
                ]
            )
        targets = {identifier: targets[identifier] for identifier in selected}

    member_count = sum(label == 1 for label in targets.values())
    nonmember_count = sum(label == 0 for label in targets.values())
    if not member_count or not nonmember_count:
        raise ValueError("Target set must contain both member and non-member examples")

    print(
        f"Preflight passed: shot={args.shot}, {len(targets)} targets "
        f"({member_count} member, {nonmember_count} non-member)"
    )
    return directory, targets


def load_existing(path: Path, args: argparse.Namespace) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames != OUTPUT_FIELDS:
            raise ValueError(f"Output schema mismatch: {path}; use --overwrite")
        for row in reader:
            if int(row["shot"]) != args.shot:
                raise ValueError(f"Output shot mismatch: {path}; use --overwrite")
            if not math.isclose(float(row["tau"]), args.wpmia_tau):
                raise ValueError(f"Output tau mismatch: {path}; use --overwrite")
            if row["embedding_model"] != args.embedding_model:
                raise ValueError(
                    f"Output embedding model mismatch: {path}; use --overwrite"
                )
            if not math.isclose(float(row["prefix_ratio"]), args.prefix_ratio):
                raise ValueError(
                    f"Output prefix ratio mismatch: {path}; use --overwrite"
                )
            expected_limit = "" if args.sample_count_limit is None else str(
                args.sample_count_limit
            )
            if row["sample_count_limit"] != expected_limit:
                raise ValueError(
                    f"Output sample-count limit mismatch: {path}; use --overwrite"
                )
            rows[row["target_id"]] = row
    return rows


def process_records(
    directory: Path,
    targets: dict[str, int],
    embedder,
    process_record,
    args: argparse.Namespace,
) -> None:
    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    completed = load_existing(output_path, args)
    missing_ids = set(targets) - set(completed)
    if not missing_ids:
        print(f"Reusing complete result: {output_path}")
        return

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=OUTPUT_FIELDS)
        if mode == "w":
            writer.writeheader()

        processed = 0
        for record in jsonl_rows(directory / "records.jsonl"):
            label = int(record["label"])
            identifier = target_id(record["input"], label)
            if identifier not in missing_ids:
                continue
            if targets[identifier] != label:
                raise ValueError(f"Label mismatch in records cache: {identifier}")

            missing_fields = REQUIRED_FIELDS - set(record)
            if missing_fields:
                raise ValueError(
                    f"Target {identifier} is missing fields: {sorted(missing_fields)}"
                )

            processed_record = process_record(
                args.gpu_id,
                SimpleNamespace(
                    wpmia_tau=args.wpmia_tau,
                    prefix_ratio=args.prefix_ratio,
                    exact_match_number=args.exact_match_number,
                    sample_count_limit=args.sample_count_limit,
                ),
                embedder,
                record,
            )
            L0 = float(processed_record["wpmia_L0"])
            Lm = float(processed_record["wpmia_Lm"])
            Lnm = float(processed_record["wpmia_Lnm"])
            if not np.all(np.isfinite([L0, Lm, Lnm])):
                raise ValueError(f"Non-finite likelihood for target {identifier}")

            row = {
                "shot": args.shot,
                "target_id": identifier,
                "target_label": label,
                "L0": f"{L0:.12g}",
                "Lm": f"{Lm:.12g}",
                "Lnm": f"{Lnm:.12g}",
                "delta_m": f"{Lm - L0:.12g}",
                "delta_nm": f"{Lnm - L0:.12g}",
                "tau": args.wpmia_tau,
                "embedding_model": args.embedding_model,
                "prefix_ratio": args.prefix_ratio,
                "sample_count_limit": ""
                if args.sample_count_limit is None
                else args.sample_count_limit,
                "cache_path": str(directory),
            }
            writer.writerow(row)
            fout.flush()
            completed[identifier] = {key: str(value) for key, value in row.items()}
            processed += 1
            if processed % 25 == 0 or not (missing_ids - set(completed)):
                print(
                    f"Processed {len(set(completed) & set(targets))}/"
                    f"{len(targets)} targets"
                )

    still_missing = set(targets) - set(completed)
    if still_missing:
        raise ValueError(f"Result is missing {len(still_missing)} targets")
    print(f"Saved {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the WPMIA prefix-mechanism target-level shifts."
    )
    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("ablation/WikiMIA/WikiMIA_length32/pythia-6.9b"),
    )
    parser.add_argument("--shot", type=int, default=7)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--wpmia_tau", type=float, default=0.2)
    parser.add_argument("--prefix_ratio", type=float, default=0.0)
    parser.add_argument("--sample_count_limit", type=int, default=None)
    parser.add_argument("--exact_match_number", action="store_true")
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow_model_download",
        action="store_true",
        help="allow Hugging Face network access instead of requiring a cached embedder",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path(
            "scripts_output_rebuttal/mechanism/"
            "wikimia_pythia69b_prefix_mechanism_per_target.csv"
        ),
    )
    args = parser.parse_args()

    if args.shot <= 0:
        raise ValueError("--shot must be > 0")
    if args.wpmia_tau <= 0:
        raise ValueError("--wpmia_tau must be > 0")
    if not 0 <= args.prefix_ratio <= 1:
        raise ValueError("--prefix_ratio must be between 0 and 1")
    if args.sample_count_limit is not None and args.sample_count_limit <= 0:
        raise ValueError("--sample_count_limit must be > 0")
    return args


def main() -> None:
    args = parse_args()
    directory, targets = preflight(args)

    if not args.overwrite:
        completed = load_existing(args.output_path, args)
        if set(targets) <= set(completed):
            print(f"Reusing complete result: {args.output_path}")
            return

    if not args.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from simmia.model import get_embedder
    from simmia.postprocess.process_word import process_wpmia_word_data

    print(f"Loading embedding model: {args.embedding_model}")
    embedder = get_embedder(
        args.gpu_id, SimpleNamespace(embedding_model=args.embedding_model)
    )
    process_records(
        directory,
        targets,
        embedder,
        process_wpmia_word_data,
        args,
    )


if __name__ == "__main__":
    main()
