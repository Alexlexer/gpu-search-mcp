"""Tests for POST /index/root and GET /index/status HTTP endpoints."""
import sys
import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post_http(path: str, payload: dict, timeout: int = 30) -> tuple[int, dict]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=timeout)
        conn.request(
            "POST",
            path,
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get_http(path: str, timeout: int = 30) -> tuple[int, dict]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=timeout)
        conn.request("GET", path)
        resp = conn.getresponse()
        body = json.loads(resp.read().decode("utf-8"))
        conn.close()
        return resp.status, body
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_repo(tmp_path: Path, name: str = "repo") -> Path:
    """Create a minimal Python repository under tmp_path/<name>."""
    repo = tmp_path / name
    repo.mkdir()
    (repo / "main.py").write_text("def hello():\n    print('hello')\n", encoding="utf-8")
    (repo / "utils.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return repo


def _restore_index_state(old_roots, old_last_result):
    """Restore mcp_server global state after a test."""
    mcp_server._http_roots = old_roots
    mcp_server._last_index_result = old_last_result


# ---------------------------------------------------------------------------
# A. POST /index/root - invalid/missing directory returns 400
# ---------------------------------------------------------------------------

def test_index_root_missing_directory_returns_400():
    status, body = _post_http("/index/root", {})
    assert status == 400
    assert "directory" in body["error"]


def test_index_root_nonexistent_directory_returns_400(tmp_path: Path):
    missing = str(tmp_path / "does_not_exist")
    status, body = _post_http("/index/root", {"directory": missing})
    assert status == 400
    assert "not found" in body["error"].lower() or "Directory" in body["error"]


def test_index_root_file_path_returns_400(tmp_path: Path):
    f = tmp_path / "notadir.py"
    f.write_text("x = 1\n", encoding="utf-8")
    status, body = _post_http("/index/root", {"directory": str(f)})
    assert status == 400
    assert "directory" in body["error"].lower() or "Not a directory" in body["error"]


# ---------------------------------------------------------------------------
# B. POST /index/root - valid temp repo returns ok, pattern ready, files > 0
# ---------------------------------------------------------------------------

def test_index_root_valid_repo_returns_ok(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        status, body = _post_http("/index/root", {"directory": str(repo)})
        assert status == 200, f"Expected 200, got {status}: {body}"
        assert body["ok"] is True
        assert body["started"] is True
        assert body["completed"] is True
        assert body["pattern"]["ready"] is True
        assert body["pattern"]["files"] > 0
        assert "fromCache" in body["pattern"]
        assert "ready" in body["dependency"]
        assert body["semantic"]["requested"] is False
        assert isinstance(body["message"], str)
        assert body["normalizedDirectory"]
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# C. After indexing, GET /diagnostics shows indexed root
# ---------------------------------------------------------------------------

def test_index_root_diagnostics_reflects_new_root(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        # Index the repo directly (not via HTTP server) so state is shared
        mcp_server._index_root(str(repo))
        _, diag = _get_http("/diagnostics")
        root_paths = [r["path"] for r in diag.get("indexedRoots", [])]
        repo_resolved = str(Path(str(repo)).resolve())
        assert any(
            repo_resolved.lower() in p.lower() or p.lower() in repo_resolved.lower()
            for p in root_paths
        ), f"Expected {repo_resolved} in {root_paths}"
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# D. After indexing, GET /stats shows file count > 0
# ---------------------------------------------------------------------------

def test_index_root_stats_shows_file_count(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        mcp_server._index_root(str(repo))
        _, stats = _get_http("/stats")
        assert stats["pattern"]["files"] > 0
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# E. After indexing, POST /scan/signals works and returns signals list
# ---------------------------------------------------------------------------

def test_index_root_scan_signals_works(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        mcp_server._index_root(str(repo))
        status, body = _post_http("/scan/signals", {"topKPerSignal": 3, "contextMode": "compact"})
        assert status == 200
        assert "signals" in body
        assert isinstance(body["signals"], list)
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# F. Root switching: index RepoA, then index RepoB, scan returns only RepoB
# ---------------------------------------------------------------------------

def test_index_root_switching_replaces_active_root(tmp_path: Path):
    repo_a = tmp_path / "repoA"
    repo_a.mkdir()
    # Use a very distinctive token unlikely to appear in repo_b
    (repo_a / "alpha.py").write_text(
        "UNIQUE_TOKEN_ALPHA_XYZZY = True\n", encoding="utf-8"
    )

    repo_b = tmp_path / "repoB"
    repo_b.mkdir()
    (repo_b / "beta.py").write_text(
        "UNIQUE_TOKEN_BETA_QQQQ = True\n", encoding="utf-8"
    )

    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        # Index A first
        result_a = mcp_server._index_root(str(repo_a))
        assert result_a["ok"], "RepoA indexing failed"
        assert mcp_server._http_roots == [str(repo_a)]

        # Index B — should replace A
        result_b = mcp_server._index_root(str(repo_b))
        assert result_b["ok"], "RepoB indexing failed"
        assert mcp_server._http_roots == [str(repo_b)]

        # Pattern search for the B token should find it
        results_b = mcp_server.index.search("UNIQUE_TOKEN_BETA_QQQQ")
        assert results_b, "Expected results for repoB token"

        # Verify active roots now only contain repoB
        roots = mcp_server._http_roots
        assert str(repo_b) in roots
        assert str(repo_a) not in roots
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# G. Cache exclusion: .gpu-search-cache files not returned after indexing
# ---------------------------------------------------------------------------

def test_index_root_excludes_gpu_search_cache_files(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    # Create a fake cache file with a distinctive signal
    cache_dir = repo / ".gpu-search-cache"
    cache_dir.mkdir()
    (cache_dir / "files-v1.json").write_text(
        '{"FAKE_CACHE_SIGNAL_DO_NOT_RETURN": true}\n', encoding="utf-8"
    )

    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        mcp_server._index_root(str(repo))
        results = mcp_server.index.search("FAKE_CACHE_SIGNAL_DO_NOT_RETURN")
        # Any results that do exist should not come from the cache directory
        for r in results:
            assert ".gpu-search-cache" not in r["file"], (
                f"Cache file appeared in results: {r['file']}"
            )
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# H. GET /index/status returns expected shape
# ---------------------------------------------------------------------------

def test_get_index_status_returns_expected_shape(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        mcp_server._index_root(str(repo))
        status, body = _get_http("/index/status")
        assert status == 200
        assert "indexedRoots" in body
        assert isinstance(body["indexedRoots"], list)
        assert "pattern" in body
        assert "ready" in body["pattern"]
        assert "files" in body["pattern"]
        assert "dependency" in body
        assert "ready" in body["dependency"]
        assert "semantic" in body
        assert "ready" in body["semantic"]
        assert "status" in body
        assert "lastIndexResult" in body
        # After indexing, lastIndexResult should be populated
        assert body["lastIndexResult"] is not None
        assert body["lastIndexResult"]["ok"] is True
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# I. includeSemantic=false result has semantic.requested=false
# ---------------------------------------------------------------------------

def test_index_root_include_semantic_false_by_default(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        status, body = _post_http(
            "/index/root", {"directory": str(repo), "includeSemantic": False}
        )
        assert status == 200
        assert body["semantic"]["requested"] is False
        assert body["semantic"]["ready"] is False
    finally:
        _restore_index_state(old_roots, old_last)


# ---------------------------------------------------------------------------
# J. includeSemantic=true with no cache returns controlled message (no exception)
# ---------------------------------------------------------------------------

def test_index_root_include_semantic_true_no_cache_no_exception(tmp_path: Path):
    repo = _make_small_repo(tmp_path)
    old_roots = list(mcp_server._http_roots)
    old_last = mcp_server._last_index_result
    try:
        status, body = _post_http(
            "/index/root", {"directory": str(repo), "includeSemantic": True}
        )
        # Should still return 200/500 based on pattern index, not crash
        assert status in (200, 500)
        assert body["semantic"]["requested"] is True
        # No cache available => ready is False and message explains it
        assert isinstance(body["semantic"]["message"], str)
        assert body["semantic"]["message"]  # non-empty
        # No unhandled exception (if we got here, it didn't crash)
    finally:
        _restore_index_state(old_roots, old_last)
