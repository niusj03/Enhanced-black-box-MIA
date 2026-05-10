import os
import json
import numpy as np
import random
import logging
import argparse
import re
from datasets import load_dataset
from typing import List

from simmia.utils import load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


def load_WikiMIA_25(sub_dataset: str) -> List[dict]:
    original_dataset = load_dataset(
        "SimMIA/WikiMIA-25", split=f"{sub_dataset}", trust_remote_code=True
    )
    original_dataset = original_dataset.to_list()
    return original_dataset


def load_WikiMIA(sub_dataset: str) -> List[dict]:
    ds = load_dataset("swj0419/WikiMIA", split=f"{sub_dataset}", trust_remote_code=True)

    member_inputs = []
    nonmember_inputs = []

    for row in ds:
        label = row["label"]
        if label == 1:
            member_inputs.append({"input": row["input"], "label": 1})
        else:
            nonmember_inputs.append({"input": row["input"], "label": 0})

    combined = member_inputs + nonmember_inputs
    return combined


def load_MIMIR(sub_dataset: str, split: str) -> List[dict]:
    ds = load_dataset(
        "iamgroot42/mimir", sub_dataset, split=split, trust_remote_code=True
    )
    member = ds["member"]
    nonmember = ds["nonmember"]

    member_data = [{"input": text, "label": 1} for text in member]
    nonmember_data = [{"input": text, "label": 0} for text in nonmember]

    combined = member_data + nonmember_data

    return combined


def get_prefix_text(prefix: str, prefix_len: int) -> str:
    words = re.findall(r"\S+", prefix)
    if len(words) >= prefix_len:
        return " ".join(words[:prefix_len])
    else:
        return " ".join(words)


def process_prefix(prefix: List[str], prefix_len: int, total_shots: int) -> List[str]:
    process_prefix_list = []
    if prefix_len > 1:
        for p in prefix:
            process_prefix_list.append(get_prefix_text(p, prefix_len))
    else:
        process_prefix_list = prefix
    process_prefix_list = process_prefix_list[:total_shots]
    return process_prefix_list


def process_data(args: argparse.Namespace, output_dir: str) -> List[dict]:
    full_path = os.path.join(output_dir, "full_dataset.jsonl")
    prefix_path = os.path.join(output_dir, "prefix_data.json")

    if all(os.path.exists(p) for p in [full_path, prefix_path]):
        # Load pre-saved datasets
        full_data = load_jsonl(full_path)
        with open(prefix_path.replace(".jsonl", ".json"), "r", encoding="utf-8") as f:
            prefix = json.load(f)

        member_data_prefix = process_prefix(
            prefix["member"], args.prefix_len, args.num_shots
        )
        nonmember_data_prefix = process_prefix(
            prefix["nonmember"], args.prefix_len, args.num_shots
        )
        member_prefix_text = "".join(member_data_prefix)
        nonmember_prefix_text = "".join(nonmember_data_prefix)
        logger.info(f"prefix is {nonmember_prefix_text}")
        logger.info("Loaded existing member, nonmember, and full datasets.")

    else:
        if "WikiMIA" in args.data:
            shots = 7
            if args.num_shots > shots:
                shots = args.num_shots
            if "-25" in args.data:
                original_dataset = load_WikiMIA_25(args.sub_dataset)
            else:
                original_dataset = load_WikiMIA(args.sub_dataset)
            logger.info("save full data and prefix")

        elif "mimir" in args.data:
            shots = 10
            if args.num_shots > shots:
                shots = args.num_shots
            original_dataset = load_MIMIR(args.sub_dataset, args.split)
            logger.info("save full data and prefix")

        else:
            raise NotImplementedError(f"Unknown `--data {args.data}`")

        # original_dataset = original_dataset.tolist()

        nonmember_data = []
        member_data = []
        for data in original_dataset:
            if data["label"] == 0:
                nonmember_data.append(data["input"])
            else:
                member_data.append(data["input"])
        logger.info(
            f"there is {len(member_data)} member data and {len(nonmember_data)} nonmember data"
        )

        random.shuffle(nonmember_data)
        nonmember_prefix = nonmember_data[:shots]
        nonmember_data = nonmember_data[shots:]

        random.shuffle(member_data)
        member_prefix = member_data[:shots]
        member_data = member_data[shots:]

        full_data = []
        for nm_data, m_data in zip(nonmember_data, member_data):
            full_data.append({"input": nm_data, "label": 0})
            full_data.append({"input": m_data, "label": 1})

        member_data_prefix = process_prefix(
            member_prefix, args.prefix_len, args.num_shots
        )
        nonmember_data_prefix = process_prefix(
            nonmember_prefix, args.prefix_len, args.num_shots
        )
        member_prefix_text = "".join(member_data_prefix)
        nonmember_prefix_text = "".join(nonmember_data_prefix)
        logger.info(f"prefix is {nonmember_prefix_text}")

        for row in full_data:
            row["prefix"] = nonmember_prefix_text

        # Save for future use
        with open(prefix_path.replace(".jsonl", ".json"), "w", encoding="utf-8") as f:
            json.dump({"member": member_prefix, "nonmember": nonmember_prefix}, f)
        save_jsonl(full_data, full_path)
        logger.info("Saved member, nonmember, and full datasets.")
    return full_data
