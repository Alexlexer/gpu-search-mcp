"""Cheap in-process usage counters for the HTTP API and the TUI dashboard.

The counters answer the question "is anyone using this server, and how" without
adding a telemetry dependency: requests and errors per endpoint, search calls
per engine, and a bounded latency sample window. Everything is process-local and
resets when the server restarts.

Threaded HTTP handlers update the collector concurrently, so every mutation and
the snapshot happen under one lock. The snapshot never allocates O(requests)
memory: latency samples are capped by ``max_samples``.
"""
from __future__ import annotations

from collections import deque
import threading
import time
from typing import Deque

DEFAULT_MAX_SAMPLES = 256


class UsageMetrics:
    """Thread-safe counters for served requests and searches."""

    def __init__(self, max_samples: int = DEFAULT_MAX_SAMPLES):
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._requests = 0
        self._errors = 0
        self._searches = 0
        self._by_endpoint: dict[str, int] = {}
        self._by_engine: dict[str, int] = {}
        self._samples_ms: Deque[float] = deque(maxlen=max_samples)
        self._max_samples = max_samples
        self._last_request: dict | None = None

    def record_request(self, endpoint: str, status: int, seconds: float) -> None:
        """Record one completed request; ``status`` >= 400 counts as an error."""
        milliseconds = max(0.0, float(seconds) * 1000.0)
        with self._lock:
            self._requests += 1
            if status >= 400:
                self._errors += 1
            endpoint = endpoint or "/"
            self._by_endpoint[endpoint] = self._by_endpoint.get(endpoint, 0) + 1
            self._samples_ms.append(milliseconds)
            self._last_request = {
                "endpoint": endpoint,
                "status": int(status),
                "milliseconds": round(milliseconds, 3),
            }

    def record_search(self, engine: str) -> None:
        """Record one search call by engine name (``search_code``, ...)."""
        with self._lock:
            self._searches += 1
            engine = engine or "unknown"
            self._by_engine[engine] = self._by_engine.get(engine, 0) + 1

    def reset(self) -> None:
        """Drop every counter and sample; used by tests."""
        with self._lock:
            self._started_at = time.time()
            self._requests = 0
            self._errors = 0
            self._searches = 0
            self._by_endpoint.clear()
            self._by_engine.clear()
            self._samples_ms.clear()
            self._last_request = None

    def snapshot(self) -> dict:
        """Return a JSON-serialisable view; safe to call from any thread."""
        with self._lock:
            samples = list(self._samples_ms)
            requests = self._requests
            errors = self._errors
            searches = self._searches
            by_endpoint = dict(self._by_endpoint)
            by_engine = dict(self._by_engine)
            last_request = dict(self._last_request) if self._last_request else None
            uptime = max(0.0, time.time() - self._started_at)
        return {
            "uptimeSeconds": round(uptime, 3),
            "requests": requests,
            "errors": errors,
            "searches": searches,
            "byEndpoint": by_endpoint,
            "bySearchEngine": by_engine,
            "latencyMs": _latency_summary(samples),
            "samplesMs": samples[-120:],
            "maxSamples": self._max_samples,
            "lastRequest": last_request,
        }


def _latency_summary(samples: list[float]) -> dict:
    if not samples:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "p50": round(_percentile(ordered, 0.50), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    """Nearest-rank percentile over an already sorted list."""
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


# One collector per process: the HTTP server writes, the TUI and /stats read.
USAGE = UsageMetrics()
