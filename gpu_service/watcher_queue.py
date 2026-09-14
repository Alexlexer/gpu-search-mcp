"""One lazy worker, bounded pending changes, no thread per filesystem event."""
import logging
import threading
import time


class Debouncer:
    def __init__(self, delay=2.0, max_pending=256):
        if delay < 0 or max_pending < 1:
            raise ValueError('Invalid debounce limits')
        self._delay = delay
        self._max_pending = max_pending
        self._pending = {}
        self._condition = threading.Condition()
        self._closed = False
        self._worker = None
        self._submitted = self._coalesced = self._completed = self._failed = 0
        self._running = 0
        self._high_water = 0
        self._waits = 0

    def stats(self):
        """Source-free counters; does not create a worker or retain event history."""
        with self._condition:
            return dict(pending=len(self._pending), capacity=self._max_pending,
                        running=self._running, submitted=self._submitted,
                        coalesced=self._coalesced, completed=self._completed,
                        failed=self._failed, high_water=self._high_water,
                        backpressure_waits=self._waits, closed=self._closed,
                        worker_alive=self._worker is not None and self._worker.is_alive())

    def submit(self, key, fn, *args):
        with self._condition:
            # Apply backpressure instead of dropping source changes or growing
            # an unbounded executor queue. Callbacks must not submit recursively.
            while not self._closed and key not in self._pending and len(self._pending) >= self._max_pending:
                self._waits += 1
                self._condition.wait()
            if self._closed:
                return
            self._submitted += 1
            self._coalesced += int(key in self._pending)
            self._pending[key] = (time.monotonic() + self._delay, fn, args)
            self._high_water = max(self._high_water, len(self._pending))
            if self._worker is None:
                self._worker = threading.Thread(target=self._run, name='gpu-search-watch-updates', daemon=True)
                self._worker.start()
            self._condition.notify_all()

    def _run(self):
        while True:
            with self._condition:
                while not self._closed and not self._pending:
                    self._condition.wait()
                if self._closed:
                    return
                key = min(self._pending, key=lambda k: self._pending[k][0])
                due, fn, args = self._pending[key]
                remaining = due - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                del self._pending[key]
                self._running = 1
                self._condition.notify_all()
            try:
                fn(*args)
            except Exception:
                with self._condition:
                    self._failed += 1
                logging.getLogger(__name__).exception('File-index update failed')
            finally:
                with self._condition:
                    self._completed += 1
                    self._running = 0

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=2.0)
