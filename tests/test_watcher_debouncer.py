"""Regression tests for bounded watchdog update scheduling."""
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server


def _scheduler_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == "gpu-search-watch-updates"
    ]


def test_debouncer_burst_uses_one_worker_thread():
    baseline = len(_scheduler_threads())
    debouncer = mcp_server._Debouncer(delay=60.0)
    try:
        for index in range(2_000):
            assert debouncer.submit(f"file:{index}", lambda: None)

        assert debouncer.pending_count == 2_000
        assert len(_scheduler_threads()) == baseline + 1
    finally:
        debouncer.close()

    assert len(_scheduler_threads()) == baseline


def test_debouncer_keeps_only_latest_callback_per_key():
    debouncer = mcp_server._Debouncer(delay=0.02)
    called: list[str] = []
    completed = threading.Event()
    try:
        debouncer.submit("same-file", called.append, "stale")
        debouncer.submit(
            "same-file",
            lambda: (called.append("latest"), completed.set()),
        )

        assert completed.wait(1.0)
        assert called == ["latest"]
        assert debouncer.pending_count == 0
    finally:
        debouncer.close()


def test_debouncer_survives_callback_failure():
    debouncer = mcp_server._Debouncer(delay=0.01)
    completed = threading.Event()

    def fail():
        raise RuntimeError("expected test failure")

    try:
        debouncer.submit("first", fail)
        # Give the first callback the earlier deadline so the regression verifies
        # that its exception does not terminate the single scheduler worker.
        time.sleep(0.002)
        debouncer.submit("second", completed.set)

        assert completed.wait(1.0)
    finally:
        debouncer.close()