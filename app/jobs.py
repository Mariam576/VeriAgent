"""Work that happens outside a request: background jobs and periodic upkeep.

Concept - why not do everything inside the request?
    A request should answer quickly. Reading a forty-page PDF and creating an
    embedding for every chunk can take a while, and some hosting platforms cut
    a request off after about a minute. So the upload endpoint can answer at
    once with "processing" and let the work go on afterwards; the interface
    asks again until the document is "ready".

Concept - in-process jobs, and what they cost:
    A job here is an asyncio task inside the API process. That needs no extra
    software, which fits one small server. The price: a job is lost when the
    process stops. VeriAgent makes that harmless instead of impossible - the
    document row says "processing", and the periodic upkeep marks rows that
    stayed in that state too long as failed, so the user sees it and can
    upload again.

    With several API processes, or work that must never be lost, the next step
    is a queue that lives outside the process (a table read with
    ``SELECT ... FOR UPDATE SKIP LOCKED``, or Redis with a worker).

A limit on jobs running at the same time protects the server's memory and the
embedding provider's rate limit.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

Job = Callable[[], Awaitable[None]]


class BackgroundJobs:
    def __init__(self, max_concurrent: int = 2) -> None:
        self._max_concurrent = max_concurrent
        self._limit: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # asyncio only keeps a weak reference to a task: without this set a job could be
        # garbage-collected in the middle of its work.
        self._tasks: set[asyncio.Task[None]] = set()
        self._periodic: set[asyncio.Task[None]] = set()

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._limit is None or self._loop is not loop:  # a new event loop (tests): start fresh
            self._limit, self._loop = asyncio.Semaphore(self._max_concurrent), loop
        return self._limit

    @property
    def active(self) -> int:
        return sum(not task.done() for task in self._tasks)

    def names(self) -> set[str]:
        """The names of the jobs that are waiting or running right now."""
        return {task.get_name().removeprefix("job:") for task in self._tasks if not task.done()}

    def submit(self, name: str, job: Job) -> None:
        """Start ``job`` after the current request. A failing job is logged, never raised."""
        limit = self._semaphore()

        async def run() -> None:
            async with limit:
                started = time.perf_counter()
                try:
                    await job()
                    logger.info("job_finished name=%s seconds=%.1f", name, time.perf_counter() - started)
                except asyncio.CancelledError:
                    logger.warning("job_cancelled name=%s", name)
                    raise
                except Exception:  # noqa: BLE001 - a job must record its own failure; this is the last line of defence
                    logger.exception("job_failed name=%s", name)

        task = asyncio.create_task(run(), name=f"job:{name}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def every(self, name: str, seconds: float, job: Job, *, first_after: float | None = None) -> None:
        """Run ``job`` again and again, ``seconds`` apart, until shutdown."""

        async def loop() -> None:
            await asyncio.sleep(seconds if first_after is None else first_after)
            while True:
                try:
                    await job()
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 - upkeep must keep going after a bad round
                    logger.exception("periodic_job_failed name=%s", name)
                await asyncio.sleep(seconds)

        task = asyncio.create_task(loop(), name=f"periodic:{name}")
        self._periodic.add(task)
        task.add_done_callback(self._periodic.discard)

    async def wait_idle(self, timeout: float = 60.0) -> bool:
        """Wait until every submitted job has finished. Returns False if that took longer than ``timeout``."""
        pending = [task for task in self._tasks if not task.done()]
        if not pending:
            return True
        _, still_running = await asyncio.wait(pending, timeout=timeout)
        return not still_running

    async def shutdown(self, timeout: float = 15.0) -> None:
        """Stop the periodic jobs, give running jobs a moment to finish, then cancel what is left."""
        for task in list(self._periodic):
            task.cancel()
        if not await self.wait_idle(timeout):
            logger.warning("Shutting down with %d job(s) still running; they are cancelled.", self.active)
            for task in list(self._tasks):
                task.cancel()
        await asyncio.gather(*self._periodic, *self._tasks, return_exceptions=True)


_jobs: BackgroundJobs | None = None


def get_jobs() -> BackgroundJobs:
    global _jobs
    if _jobs is None:
        from app.config import get_settings

        _jobs = BackgroundJobs(get_settings().MAX_BACKGROUND_JOBS)
    return _jobs
