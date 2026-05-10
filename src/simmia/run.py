import argparse
import os
import multiprocessing
import logging
import sys
import json

import numpy as np
import random
import torch
from transformers import set_seed

from pathlib import Path
from ast import literal_eval

from simmia.options import add_arguments
from simmia._internal import (
    SAMPLING_METHODS,
    POSTPROCESS_METHODS,
    INFERENCE_METHODS,
    ASYNC_SAMPLING_METHODS,
)
from simmia.benchmark import process_data
from simmia.model import get_model, get_embedder
from simmia.utils import evaluate
from simmia.parallel import JobScheduler, AsyncJobScheduler


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def cli_main():
    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.params is not None:
        for kv in args.params.split(","):
            key, value = kv.split(":")
            try:
                args.__dict__[key.strip()] = literal_eval(value.strip())
            except:
                args.__dict__[key.strip()] = value.strip()

    fix_seed(args.seed)

    if "WikiMIA" in args.data:
        output_dir = f"{args.output_dir}/{os.path.split(args.data)[-1]}/{args.sub_dataset}/{os.path.split(args.model_name_or_path)[-1]}/{args.num_shots}"
    elif "mimir" in args.data:
        output_dir = f"{args.output_dir}/{os.path.split(args.data)[-1]}/{args.split}/{os.path.split(args.model_name_or_path)[-1]}/{args.sub_dataset}/{args.num_shots}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outpath = Path(output_dir) / "records.jsonl"
    if os.path.exists(outpath) and not args.overwrite:
        # Load previous cache if it exists / the user does not want to overwrite the cache
        logger.info(f"Loading results from {Path(outpath).cwd() / outpath}")
        with open(outpath, "r", encoding="utf-8") as fin:
            records = [json.loads(line.strip()) for line in fin.readlines()]
    else:
        records = []

    full_data = process_data(
        args,
        output_dir,
    )

    if len(records) < len(full_data):
        logger.info(f"Saving results to {Path(outpath).cwd() / outpath}")
        with open(outpath, "a", encoding="utf-8") as fin:
            # Run & write cache
            scheduler_cls = (
                AsyncJobScheduler
                if args.model_name_or_path.startswith("api:")
                else JobScheduler
            )
            sampling_methods = (
                ASYNC_SAMPLING_METHODS
                if args.model_name_or_path.startswith("api:")
                else SAMPLING_METHODS
            )
            with scheduler_cls(
                args=args,
                job_init_fn=get_model,
                job_exec_fn=sampling_methods[args.sampling],
                devices=args.gpu_ids,
                concurrency=args.concurrency,
            ) as scheduler:
                for record in scheduler.execute_jobs(
                    jobs=full_data[len(records) :],
                    desc=f"Sampling with [{sampling_methods[args.sampling].__name__}]",
                ):
                    records.append(record)
                    fin.write(json.dumps(record) + "\n")

    # Postprocess
    if args.postprocess is not None:
        jobs = records
        records = []
        with JobScheduler(
            args=args,
            job_init_fn=get_embedder,
            job_exec_fn=POSTPROCESS_METHODS[args.postprocess],
            devices=args.gpu_ids,
        ) as scheduler:
            for record in scheduler.execute_jobs(
                jobs=jobs,
                desc=f"Postprocessing with [{POSTPROCESS_METHODS[args.postprocess].__name__}]",
            ):
                records.append(record)

    # Evaluation
    predictions, answers = INFERENCE_METHODS[args.inference](args, records[:])

    auc, acc, low1, low5, low10 = evaluate(
        predictions,
        answers,
        output_dir,
    )
    if args.data == "iamgroot42/mimir":
        dataset_name = f"MIMIR/{args.split}/{args.sub_dataset}"
    elif args.data == "swj0419/WikiMIA":
        dataset_name = f"WikiMIA/{args.sub_dataset}"
    elif args.data == "SimMIA/WikiMIA-25":
        dataset_name = f"WikiMIA-25/{args.sub_dataset}"
    else:
        raise ValueError(f"Unknown dataset: {args.data}")
    logger.info(
        f"{dataset_name}: AUC {auc*100:.1f}, Accuracy {acc*100:.1f}, TPR@1%FPR of {low1*100:.1f}, TPR@5%FPR of {low5*100:.1f}, TPR@10%FPR of {low10*100:.1f}"
    )
