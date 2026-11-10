"""Tests for the in-process usage counters behind /stats and the TUI."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from usage_stats import UsageMetrics


def test_counters_start_empty():
    metrics = UsageMetrics().snapshot()

    assert metrics["requests"] == 0
    assert metrics["errors"] == 0
    assert metrics["searches"] == 0
    assert metrics["latencyMs"]["count"] == 0
    assert metrics["byEndpoint"] == {}


def test_record_request_tracks_endpoint_status_and_latency():
    metrics = UsageMetrics()
    metrics.record_request("/search/code", 200, 0.01)
    metrics.record_request("/search/code", 200, 0.03)
    metrics.record_request("/read/block", 500, 0.02)

    snapshot = metrics.snapshot()

    assert snapshot["requests"] == 3
    assert snapshot["errors"] == 1
    assert snapshot["byEndpoint"] == {"/search/code": 2, "/read/block": 1}
    assert snapshot["latencyMs"]["count"] == 3
    assert snapshot["latencyMs"]["max"] == pytest.approx(30.0)
    assert snapshot["lastRequest"] == {
        "endpoint": "/read/block",
        "status": 500,
        "milliseconds": 20.0,
    }


def test_latency_summary_is_ordered_and_bounded():
    metrics = UsageMetrics(max_samples=5)
    for milliseconds in (10, 20, 30, 40, 50, 60, 70):
        metrics.record_request("/x", 200, milliseconds / 1000.0)

    snapshot = metrics.snapshot()

    assert snapshot["maxSamples"] == 5
    assert len(snapshot["samplesMs"]) == 5
    values = snapshot["samplesMs"]
    assert min(values) == pytest.approx(30.0)
    assert max(values) == pytest.approx(70.0)
    assert snapshot["latencyMs"]["p50"] <= snapshot["latencyMs"]["p95"]


def test_record_search_counts_per_engine():
    metrics = UsageMetrics()
    metrics.record_search("search_code")
    metrics.record_search("search_code")
    metrics.record_search("search_hybrid")

    snapshot = metrics.snapshot()

    assert snapshot["searches"] == 3
    assert snapshot["bySearchEngine"] == {"search_code": 2, "search_hybrid": 1}


def test_snapshot_is_a_copy_and_reset_clears_everything():
    metrics = UsageMetrics()
    metrics.record_request("/x", 200, 0.001)
    metrics.record_search("search_code")

    snapshot = metrics.snapshot()
    snapshot["byEndpoint"]["/x"] = 999

    assert metrics.snapshot()["byEndpoint"]["/x"] == 1

    metrics.reset()
    cleared = metrics.snapshot()
    assert cleared["requests"] == 0
    assert cleared["searches"] == 0
    assert cleared["lastRequest"] is None


def test_invalid_sample_limit_is_rejected():
    with pytest.raises(ValueError):
        UsageMetrics(max_samples=0)
