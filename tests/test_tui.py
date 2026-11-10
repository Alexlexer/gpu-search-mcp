"""Tests for the TUI snapshot collection, formatting and rendering."""
from __future__ import annotations

import socket
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server
import tui


def _snapshot() -> dict:
    return {
        "url": "http://127.0.0.1:8765",
        "fetchedAt": 1_700_000_000.0,
        "error": None,
        "warning": None,
        "stats": {
            "version": "0.1.0",
            "device": {"backend": "cpu"},
            "pattern": {
                "files": 1234,
                "chunks": 233,
                "corpus_bytes": 134_217_728,
                "cache": "loaded",
                "candidate_selector": "TrigramCandidateSelector",
                "storage_backend": "FileStorageBackend",
                "chunk_size": 2 * 1024 * 1024,
                "buffer_count": 2,
                "base_dir": "C:/repo",
                "candidate_index": {"indexed_keys": 48000, "index_bytes": 3_100_000},
                "last_query": {
                    "number_of_chunks": 233,
                    "candidate_chunks": 2,
                    "candidate_percentage": 0.86,
                    "bytes_read_from_storage": 4_300_000,
                    "corpus_percentage_physically_read": 3.2,
                    "total_query_seconds": 0.0032,
                    "storage_read_seconds": 0.0009,
                    "gpu_search_seconds": 0.0011,
                    "host_to_gpu_seconds": 0.0004,
                    "pipeline_enabled": True,
                    "prefetched_chunks": 1,
                },
            },
            "semantic": {"chunks": 9200, "vram_mb": 137.0},
            "dependency": {"files": 1180, "edges": 4821},
            "symbols": {"symbols": 3400, "edges": 5100},
            "capabilities": {"patternSearch": True, "semanticSearch": True},
            "usage": {
                "uptimeSeconds": 83.4,
                "requests": 87,
                "errors": 0,
                "searches": 12,
                "byEndpoint": {"/search/code": 70},
                "bySearchEngine": {"search_code": 12},
                "latencyMs": {"p50": 0.6, "p95": 3.1, "max": 12.4},
                "samplesMs": [0.4, 0.6, 2.1, 0.5, 3.4],
                "lastRequest": {
                    "endpoint": "/search/code",
                    "status": 200,
                    "milliseconds": 0.5,
                },
            },
        },
        "runtime": {
            "process": {
                "resident_bytes": 432_000_000,
                "peak_resident_bytes": 510_000_000,
                "python_threads": 14,
                "pid": 4242,
            },
            "pattern_buffers": {
                "host_staging_bytes": 8_388_608,
                "device_bytes": 0,
                "available_buffers": 2,
                "leased_buffers": 0,
                "max_owned_bytes": 67_108_864,
            },
            "watcher": {"pending": 0, "coalesced": 12, "completed": 30, "failed": 0},
        },
    }


def _render_text(renderable) -> str:
    from rich.console import Console

    console = Console(record=True, width=120)
    console.print(renderable)
    return console.export_text()


def _start_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, thread


def _stop_server(httpd, thread):
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def test_collect_snapshot_reads_stats_and_runtime():
    # Warm the lazy services first: the first /stats in a fresh process imports
    # torch and the semantic stack inside the handler thread, which can exceed a
    # short client timeout even though the endpoint itself is cheap afterwards.
    mcp_server.index.stats()
    mcp_server.semantic.stats()
    mcp_server.deps.stats()
    mcp_server.symbols.stats()
    mcp_server.cache_metadata_for_stats()
    mcp_server.semantic_model_status_for_stats()
    httpd, thread = _start_server()
    try:
        snapshot = tui.collect_snapshot(f"http://127.0.0.1:{httpd.server_port}", timeout=20)
    finally:
        _stop_server(httpd, thread)

    assert snapshot["error"] is None
    assert snapshot["stats"]["version"] == mcp_server.VERSION
    assert "usage" in snapshot["stats"]
    assert "process" in snapshot["runtime"]


def test_collect_snapshot_reports_unreachable_server():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    snapshot = tui.collect_snapshot(f"http://127.0.0.1:{port}", timeout=1.0)

    assert snapshot["error"]
    assert str(port) in snapshot["error"]
    assert snapshot["stats"] == {}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [(0, "0 B"), (512, "512 B"), (1536, "1.5 KiB"), (134_217_728, "128.0 MiB")],
)
def test_format_bytes(value, expected):
    assert tui.format_bytes(value) == expected


def test_format_helpers_handle_missing_values():
    assert tui.format_int(None) == tui._GLYPHS["dash"]
    assert tui.format_ms(None) == tui._GLYPHS["dash"]
    assert tui.format_percent(None) == tui._GLYPHS["dash"]
    assert tui.format_duration(None) == tui._GLYPHS["dash"]


def test_format_values():
    assert tui.format_int(1234) == "1,234"
    assert tui.format_ms(0.0032) == "3.2 ms"
    assert tui.format_percent(0.86) == "0.9%"
    assert tui.format_duration(3661) == "01:01:01"


def test_sparkline_edges():
    assert len(tui.sparkline([])) == 8
    assert len(tui.sparkline([1.0])) == 1
    rendered = tui.sparkline([1.0, 2.0, 3.0, 4.0])
    assert len(rendered) == 4
    assert set(rendered) <= set(tui._GLYPHS["spark"])


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_render_dashboard_shows_every_section():
    text = _render_text(tui.render_dashboard(_snapshot()))

    for fragment in [
        "Overview",
        "Indexes",
        "Last exact query",
        "Resources",
        "Usage (HTTP API)",
        "Trigram",
        "gpu-search",
    ]:
        assert fragment in text, fragment
    assert "1,234" in text
    assert "128.0 MiB" in text


def test_render_dashboard_connection_error_shows_hint():
    snapshot = {"url": "http://127.0.0.1:1", "fetchedAt": 0.0, "error": "URLError: refused", "stats": {}, "runtime": {}}

    text = _render_text(tui.render_dashboard(snapshot))

    assert "No server" in text
    assert "gpu-search-tui --directory" in text


def test_render_dashboard_tolerates_partial_snapshot():
    text = _render_text(tui.render_dashboard({"url": "http://x", "stats": {}, "runtime": {}}))

    assert "Overview" in text


def test_ascii_glyph_fallback_for_narrow_encodings(monkeypatch):
    class FakeStream:
        encoding = "cp1252"

    monkeypatch.setattr(tui, "_GLYPHS_READY", False)
    monkeypatch.setattr(tui, "_GLYPHS", dict(tui.UNICODE_GLYPHS))

    tui._ensure_glyphs(FakeStream())

    assert tui._GLYPHS == tui.ASCII_GLYPHS


def test_unicode_glyphs_survive_utf8(monkeypatch):
    class FakeStream:
        encoding = "utf-8"

    monkeypatch.setattr(tui, "_GLYPHS_READY", False)
    monkeypatch.setattr(tui, "_GLYPHS", dict(tui.UNICODE_GLYPHS))

    tui._ensure_glyphs(FakeStream())

    assert tui._GLYPHS == tui.UNICODE_GLYPHS


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def test_log_buffer_keeps_a_bounded_tail():
    buffer = tui.LogBuffer(max_lines=3)
    for index in range(10):
        buffer.write(f"line {index}\n")

    assert buffer.lines() == ["line 7", "line 8", "line 9"]
    assert buffer.isatty() is False


def test_parser_defaults_and_validation():
    parser = tui.build_parser()
    args = parser.parse_args([])
    assert args.url == tui.DEFAULT_URL
    assert args.directories == []
    assert args.interval == 1.0

    with pytest.raises(SystemExit):
        tui.main(["--interval", "0.01"])
    with pytest.raises(SystemExit):
        tui.main(["--directory", ".", "--url", "http://127.0.0.1:9999"])
