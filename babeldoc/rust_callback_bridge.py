"""Python-owned callback executors and shared run cancellation."""

import threading
from concurrent.futures import ThreadPoolExecutor


class ServiceExecutor:
    """Python owns the service and its executor; Rust owns each result channel."""

    def __init__(self, service, workers, cancel_flag, record_requests=True):
        self.service = service
        self.cancel_flag = cancel_flag
        self.record_requests = record_requests
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="p2-service"
        )

    def submit(self, method, args, kwargs, completion):
        def invoke():
            if method not in getattr(self.service, "cleanup_methods", frozenset()):
                self.cancel_flag.checkpoint()
            return getattr(self.service, method)(*args, **(kwargs or {}))

        future = self.pool.submit(invoke)
        # Native completion never raises; service exceptions live in Future.result().
        future.add_done_callback(completion)

    def close(self):
        self.pool.shutdown(wait=True)


class CallbackRun:
    """Own the Python threads for one run without mutating the caller's Event."""

    def __init__(self, services, cancel_event, native, record_requests=True):
        self.cancel_event = cancel_event
        self.flag = native.CancelFlag()
        self.executors = {}
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="p2-cancel-watch")
        try:
            self.add_services(services, record_requests=record_requests)
            self._watcher.start()
        except BaseException:
            self.close()
            raise

    def add_services(self, services, record_requests=True):
        """Bind services after parsing while retaining the run's cancellation flag."""
        for name, (service, workers) in services.items():
            if name in self.executors:
                raise ValueError(f"callback service is already bound: {name}")
            self.executors[name] = ServiceExecutor(
                service, workers, self.flag, record_requests
            )

    def _watch(self):
        # Event has no callback hook. Bounded waits let the owner reclaim the watcher
        # at normal exit without setting the external cancellation Event.
        while not self._stop.is_set():
            if self.cancel_event.wait(0.02):
                self.flag.cancel()
                return

    def close(self):
        try:
            for executor in reversed(tuple(self.executors.values())):
                executor.close()
        finally:
            self._stop.set()
            if self._watcher.ident is not None:
                self._watcher.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
