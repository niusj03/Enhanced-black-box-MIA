import argparse
import asyncio
import logging
import queue
import threading

from tqdm import tqdm
from typing import Callable, List, Any, Awaitable


logger = logging.getLogger(__name__)


__all__ = [
    "AsyncJobScheduler",
]


class AsyncJobScheduler:
    """
    A simple asynchronous job scheduler for fast parallel I/O intensive execution.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        job_init_fn: Callable[[int, argparse.Namespace], dict],
        job_exec_fn: Callable[[int, argparse.Namespace, dict, dict], Awaitable[dict]],
        concurrency: int,
        **kwargs,
    ):
        self.args = args
        self.job_init_fn = job_init_fn
        self.job_exec_fn = job_exec_fn
        self.concurrency = concurrency
        self.SENTINEL = object()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute_jobs(self, jobs: List[Any], desc: str = "Executing jobs"):
        """
        The main entry point for parallel execution of a list of jobs.

        Note: the output order is guaranteed to be identical to the input order.
        """
        stop_event = threading.Event()
        output_queue = queue.Queue()
        p_bar = tqdm(desc=desc, total=len(jobs))

        task = threading.Thread(
            target=self._run, args=(jobs, output_queue, stop_event), daemon=True
        )
        task.start()

        try:
            while True:
                result = output_queue.get()
                if isinstance(result, Exception):
                    raise result
                if result is self.SENTINEL:
                    break
                yield result
                p_bar.update(1)
        finally:
            stop_event.set()
            p_bar.close()
            task.join()

    def _run(
        self, jobs: List[Any], output_queue: queue.Queue, stop_event: threading.Event
    ):
        try:
            asyncio.run(self._async_run(jobs, output_queue, stop_event))
        except Exception as e:
            output_queue.put(e)
        else:
            output_queue.put(self.SENTINEL)

    async def _async_run(
        self, jobs: List[Any], output_queue: queue.Queue, stop_event: threading.Event
    ):
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = []
        for rank, job in enumerate(jobs):
            if stop_event.is_set():
                break
            task = asyncio.create_task(self._safe_exec(rank, job, semaphore))
            tasks.append(task)

            if len(tasks) > self.concurrency * 2:
                first_task = tasks.pop(0)
                try:
                    res = await first_task
                    output_queue.put(res)
                except asyncio.CancelledError:
                    pass

        if stop_event.is_set():
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            for task in tasks:
                try:
                    res = await task
                    output_queue.put(res)
                except asyncio.CancelledError:
                    pass

    async def _safe_exec(
        self,
        rank: int,
        job: dict,
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
            state = self.job_init_fn(rank, self.args)
            return await self.job_exec_fn(rank, self.args, state, job)
