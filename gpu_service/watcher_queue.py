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

    def submit(self, key, fn, *args):
        with self._condition:
            # Apply backpressure instead of dropping source changes or growing
            # an unbounded executor queue. Callbacks must not submit recursively.
            while not self._closed and key not in self._pending and len(self._pending) >= self._max_pending:
                self._condition.wait()
            if self._closed:
                return
            self._pending[key] = (time.monotonic() + self._delay, fn, args)
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
                self._condition.notify_all()
            try:
                fn(*args)
            except Exception:
                logging.getLogger(__name__).exception('File-index update failed')

    def close(self):
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._condition.notify_all()
        if self._worker and self._worker is not threading.current_thread():
            self._worker.join(timeout=2.0)
