"""Tests for the lightweight diagnostics endpoint."""
import json
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server


def _get_http(path: str) -> tuple[int, dict]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class FakeIndex:
    def __init__(self, root: str | None = None, files: int = 0):
        self.root = root
        self.files = files

    def stats(self):
        return {"files": self.files, "base_dir": self.root, "vram_mb": 1.5, "cache": "loaded"}


class FakeSemantic:
    def __init__(self, root: str | None = None, chunks: int = 0):
        self.root = root
        self.chunks = chunks

    def stats(self):
        return {"chunks": self.chunks, "base_dir": self.root, "vram_mb": 0.0}


class FakeDeps:
    def __init__(self, root: str | None = None, files: int = 0, edges: int = 0):
        self.root = root
        self.files = files
        self.edges = edges

    def stats(self):
        return {"files": self.files, "edges": self.edges, "base_dir": self.root, "cache": "loaded"}


def _patch_runtime(monkeypatch, tmp_path: Path, *, pattern_files=3, semantic_available=False):
    monkeypatch.setattr(mcp_server, "index", FakeIndex(str(tmp_path), pattern_files))
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic(str(tmp_path), 0))
    monkeypatch.setattr(mcp_server, "deps", FakeDeps(str(tmp_path), 2, 1))
    monkeypatch.setattr(mcp_server, "_http_roots", [str(tmp_path)])
    monkeypatch.setattr(
        mcp_server,
        "semantic_model_status_for_stats",
        lambda: {
            "modelId": "BAAI/bge-small-en-v1.5",
            "provider": "sentence-transformers",
            "available": semantic_available,
            "cached": semantic_available,
            "requiresDownload": not semantic_available,
            "device": "cpu",
            "message": "ok" if semantic_available else "not cached",
        },
    )
    monkeypatch.setattr(
        mcp_server,
        "cache_metadata_for_stats",
        lambda: {
            "schemaVersion": 1,
            "directory": str(tmp_path / ".gpu-search-cache"),
            "entries": [{"name": "pattern", "schemaVersion": 1, "status": "loaded"}],
        },
    )


def test_diagnostics_endpoint_returns_200(monkeypatch, tmp_path: Path):
    _patch_runtime(monkeypatch, tmp_path)

    status, body = _get_http("/diagnostics")

    assert status == 200
    assert body["version"] == mcp_server.VERSION
    assert body["status"] in {"ok", "degraded", "not_ready"}


def test_diagnostics_includes_device_metadata(monkeypatch, tmp_path: Path):
    _patch_runtime(monkeypatch, tmp_path)

    _, body = _get_http("/diagnostics")

    assert "device" in body
    assert "backend" in body["device"]
    assert "torchDevice" in body["device"]
    assert "warnings" in body["device"]


def test_diagnostics_includes_capabilities(monkeypatch, tmp_path: Path):
    _patch_runtime(monkeypatch, tmp_path)

    _, body = _get_http("/diagnostics")

    assert body["capabilities"]["patternSearch"] is True
    assert body["capabilities"]["dependencyImpact"] is True
    assert body["capabilities"]["signalScan"] is True
    assert body["capabilities"]["mcpTools"] is True


def test_diagnostics_includes_semantic_model_without_download(monkeypatch, tmp_path: Path):
    calls = {"status": 0, "download": 0}

    def fake_status():
        calls["status"] += 1
        return {
            "modelId": "BAAI/bge-small-en-v1.5",
            "provider": "sentence-transformers",
            "available": False,
            "cached": False,
            "requiresDownload": True,
            "message": "Semantic model is not available locally.",
        }

    _patch_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(mcp_server, "semantic_model_status_for_stats", fake_status)
    monkeypatch.setattr(
        mcp_server,
        "download_semantic_model",
        lambda *args, **kwargs: calls.__setitem__("download", calls["download"] + 1),
    )

    _, body = _get_http("/diagnostics")

    assert calls["status"] == 1
    assert calls["download"] == 0
    assert body["indexes"]["semantic"]["modelAvailable"] is False
    assert body["indexes"]["semantic"]["modelId"] == "BAAI/bge-small-en-v1.5"


def test_diagnostics_handles_no_indexed_root_gracefully(monkeypatch):
    monkeypatch.setattr(mcp_server, "index", FakeIndex(None, 0))
    monkeypatch.setattr(mcp_server, "semantic", FakeSemantic(None, 0))
    monkeypatch.setattr(mcp_server, "deps", FakeDeps(None, 0, 0))
    monkeypatch.setattr(mcp_server, "_http_roots", [])
    monkeypatch.setattr(
        mcp_server,
        "semantic_model_status_for_stats",
        lambda: {"modelId": "x", "available": False, "message": "missing"},
    )
    monkeypatch.setattr(
        mcp_server,
        "cache_metadata_for_stats",
        lambda: {"schemaVersion": 1, "directory": None, "entries": []},
    )

    status, body = _get_http("/diagnostics")

    assert status == 200
    assert body["status"] == "not_ready"
    assert body["indexedRoots"] == []
    assert any("No indexed roots" in w for w in body["warnings"])


def test_diagnostics_snapshot_does_not_load_semantic_model(monkeypatch, tmp_path: Path):
    class RaisingSemantic(FakeSemantic):
        def _get_model(self):
            raise AssertionError("diagnostics must not load semantic model")

    _patch_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(mcp_server, "semantic", RaisingSemantic(str(tmp_path), 0))

    body = mcp_server.diagnostics_snapshot()

    assert body["indexes"]["semantic"]["ready"] is False
