import asyncio
import itertools
import logging
import queue
import sys
import threading
from concurrent import futures

logger = logging.getLogger(__name__)

DEFAULT_PRIORITY = 1048576
_WAKEUP_PRIORITY = sys.maxsize


class _PriorityWorkQueue(queue.PriorityQueue):
    """Variant of Queue that retrieves open entries in priority order (lowest first)."""

    def _init(self, maxsize):
        super()._init(maxsize)
        self._counter = itertools.count()
        self._local = threading.local()

    def set_priority(self, priority: int) -> None:
        self._local.priority = priority

    def _put(self, item):
        if item is None:
            priority = _WAKEUP_PRIORITY
        else:
            priority = getattr(self._local, "priority", DEFAULT_PRIORITY)
        super()._put((priority, next(self._counter), item))

    def _get(self):
        return super()._get()[2]


def _log_future_exception(future: futures.Future) -> None:
    if future.cancelled():
        return
    exception = future.exception()
    if exception is None or isinstance(exception, asyncio.CancelledError):
        return
    logger.warning("Exception in future %s: %s", future, exception, exc_info=exception)


class PriorityThreadPoolExecutor(futures.ThreadPoolExecutor):
    """
    Thread pool executor with priority queue (lowest first).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._work_queue: _PriorityWorkQueue = _PriorityWorkQueue()

    __init__.__doc__ = futures.ThreadPoolExecutor.__init__.__doc__

    def submit(self, fn, /, *args, priority: int = DEFAULT_PRIORITY, **kwargs):
        """Submits a callable to be executed with the given priority and arguments.

        Schedules the callable to be executed as fn(*args, **kwargs) and
        returns a Future instance representing the execution of the callable.

        Args:
            priority: The priority of the future.

        Returns:
            A Future representing the given call.
        """
        self._work_queue.set_priority(priority)
        try:
            future = super().submit(fn, *args, **kwargs)
        finally:
            self._work_queue.set_priority(DEFAULT_PRIORITY)
        future.add_done_callback(_log_future_exception)
        return future

    def shutdown(self, wait=True, *, cancel_futures=False):
        logger.debug("%s shutting down", self._thread_name_prefix)
        super().shutdown(wait=wait, cancel_futures=cancel_futures)
        logger.debug("%s shutdown finish", self._thread_name_prefix)

    shutdown.__doc__ = futures.ThreadPoolExecutor.shutdown.__doc__
