import argparse
import transformers
import multiprocessing
import traceback
import logging

from tqdm import tqdm
from queue import PriorityQueue
from typing import Callable, List, Any, Optional


logger = logging.getLogger(__name__)


__all__ = [
    "JobScheduler",
]


class JobScheduler:
    """
    A simple multiprocessing-based job scheduler for fast parallel execution on (GPU) devices.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        job_init_fn: Callable[[int, argparse.Namespace], dict],
        job_exec_fn: Callable[[int, argparse.Namespace, dict, dict], dict],
        devices: List[int],
        **kwargs,
    ):
        self.args = args
        self.job_init_fn = job_init_fn
        self.job_exec_fn = job_exec_fn
        self.devices = devices if len(devices) else [0]

        self.input_queue: Optional[multiprocessing.Queue] = None
        self.output_queue: Optional[multiprocessing.Queue] = None

        self.process_pool: List[multiprocessing.Process] = []

    def start(self):
        """
        Acquire the resources before calling .execute_jobs
        """
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()

        for rank in self.devices:
            p = multiprocessing.Process(
                target=job_fn_wrapper,
                args=[
                    rank,
                    self.args,
                    self.input_queue,
                    self.output_queue,
                    self.job_init_fn,
                    self.job_exec_fn,
                ],
                daemon=True,
            )
            p.start()
            self.process_pool.append(p)

    def close(self):
        """
        Release the resources after calling .execute_jobs
        """
        for p in self.process_pool:
            p.terminate()
        self.input_queue.close()
        self.output_queue.close()
        self.input_queue.cancel_join_thread()
        self.output_queue.cancel_join_thread()

        self.process_pool = []
        self.input_queue = None
        self.output_queue = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            return False
        return True

    def execute_jobs(self, jobs: List[Any], desc: str = "Executing jobs"):
        """
        The main entry point for parallel execution of a list of jobs.

        Note: the output order is gauranteed to be identical to the input order.
        """
        sorted_queue = PriorityQueue()

        for priority, job in enumerate(jobs):
            self.input_queue.put((priority, job))

        output_counter = 0
        progress_bar = tqdm(desc=desc, total=len(jobs))
        while output_counter < len(jobs):
            job_result = self.output_queue.get()
            if isinstance(job_result, Exception):
                raise job_result
            priority, job_result = job_result
            sorted_queue.put((priority, job_result))
            while not sorted_queue.empty():
                top_priority, top_result = sorted_queue.get()
                if top_priority == output_counter:
                    output_counter += 1
                    yield top_result
                    progress_bar.update(1)
                else:
                    sorted_queue.put((top_priority, top_result))
                    break


def job_fn_wrapper(
    rank: int,
    args: argparse.Namespace,
    input_queue: multiprocessing.Queue,
    output_queue: multiprocessing.Queue,
    job_init_fn: Callable[[int, argparse.Namespace], dict],
    job_exec_fn: Callable[[int, argparse.Namespace, dict, dict], dict],
):
    try:
        logger.info(f"[RANK {rank}] Initializing process with `{job_init_fn.__name__}`")
        state = job_init_fn(rank, args)
        logger.info(f"[RANK {rank}] Ready to run process with `{job_exec_fn.__name__}`")
    except Exception as e:
        traceback.print_exc()
        print(f"[Rank {rank}][{type(e)}]: ", e, flush=True)
        output_queue.put(e)

    while True:
        idx, job = input_queue.get()
        if job is None:
            break

        # Fix the random seed for each sample to ensure reproducibility in multiprocessing
        sampling_seed = getattr(args, "sampling_seed", args.seed)
        if sampling_seed is not None:
            transformers.set_seed(sampling_seed)

        try:
            result = job_exec_fn(rank, args, state, job)
        except Exception as e:
            traceback.print_exc()
            print(f"[Rank {rank}][{type(e)}]: ", e, flush=True)
            output_queue.put(e)
            return

        output_queue.put((idx, result))
