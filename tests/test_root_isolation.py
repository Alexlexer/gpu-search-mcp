"""Regression tests for root isolation and cache leakage (fix/root-isolation-cache-leakage).

Covers:
  A. Root switch isolation — pattern index after switching repos
  B. Signal scan root isolation — scan does not return cross-repo paths
  C. Cache directory exclusion — .gpu-search-cache files never in results
  D. Cache root mismatch — cache written for a different directory is rejected
  E. Diagnostics correctness — indexedRoots matches active root
  F. Result boundary filter — unit test for _is_allowed_result
  G. _prepare_startup — explicit --directory suppresses config dir merging

All tests use CPU-only pattern index (no GPU / semantic model required).
"""
from __future__ import annotations

import json
import sys
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server
from http_server import _is_allowed_result, _filter_to_active_roots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_http(path: str, payload: dict) -> tuple[int, dict]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), mcp_server._HttpApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.request(
            "POST", path,
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


class _FakeIndex:
    """Lightweight in-memory index used to inject controlled search results."""

    def __init__(self, base: str, file_contents: dict[str, str]):
        self._base = base
        self._files = file_contents

    def stats(self):
        return {"files": max(1, len(self._files)), "base_dir": self._base, "vram_mb": 0}

    def search(self, query, case_sensitive=False, max_files=50):
        results = []
        for fpath, content in self._files.items():
            matches = [
                {"line": i, "content": line}
                for i, line in enumerate(content.splitlines(), start=1)
                if query in line
            ]
            if matches:
                results.append({"file": fpath, "matches": matches, "_total_files": 1})
        return results


class _FakeSemantic:
    def stats(self):
        return {"chunks": 0, "base_dir": None, "vram_mb": 0}

    def semantic_unavailable_message(self):
        return "Semantic index not ready."


class _FakeDeps:
    def stats(self):
        return {"files": 0, "edges": 0, "base_dir": None, "cache": "cold"}


# ---------------------------------------------------------------------------
# F. Result boundary filter — unit tests for _is_allowed_result
# ---------------------------------------------------------------------------

def test_boundary_filter_allows_path_under_root(tmp_path):
    child = tmp_path / "src" / "app.py"
    child.parent.mkdir(parents=True, exist_ok=True)
    child.write_text("x=1")
    assert _is_allowed_result(str(child), [str(tmp_path)])


def test_boundary_filter_rejects_path_outside_root(tmp_path):
    sibling = tmp_path.parent / "other_repo" / "app.py"
    assert not _is_allowed_result(str(sibling), [str(tmp_path)])


def test_boundary_filter_rejects_gpu_search_cache_path(tmp_path):
    cache_file = tmp_path / ".gpu-search-cache" / "files-v1.json"
    assert not _is_allowed_result(str(cache_file), [str(tmp_path)])


def test_boundary_filter_allows_when_no_roots():
    # No active roots = don't break zero-config mode
    assert _is_allowed_result("/some/path/file.py", [])


def test_boundary_filter_empty_path():
    assert _is_allowed_result("", ["/some/root"])


def test_filter_to_active_roots_removes_outside(tmp_path):
    inside = tmp_path / "a.py"
    outside = tmp_path.parent / "other" / "b.py"
    results = [
        {"absoluteFile": str(inside), "file": "a.py"},
        {"absoluteFile": str(outside), "file": "b.py"},
    ]
    filtered = _filter_to_active_roots(results, [str(tmp_path)])
    assert len(filtered) == 1
    assert filtered[0]["file"] == "a.py"


def test_filter_to_active_roots_removes_cache_path(tmp_path):
    cache_file = tmp_path / ".gpu-search-cache" / "files-v1.json"
    real_file = tmp_path / "app.py"
    results = [
        {"absoluteFile": str(cache_file), "file": ".gpu-search-cache/files-v1.json"},
        {"absoluteFile": str(real_file), "file": "app.py"},
    ]
    filtered = _filter_to_active_roots(results, [str(tmp_path)])
    assert len(filtered) == 1
    assert filtered[0]["file"] == "app.py"


# ---------------------------------------------------------------------------
# A. Root switch isolation — pattern index
# ---------------------------------------------------------------------------

def test_pattern_index_reset_on_directory_switch(tmp_path):
    """After indexing RepoB, the pattern index must not return RepoA files."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()
    (repo_a / "alpha.py").write_text("ONLY_REPO_A_SIGNAL = True\n", encoding="utf-8")

    repo_b = tmp_path / "RepoB"
    repo_b.mkdir()
    (repo_b / "beta.py").write_text("ONLY_REPO_B_SIGNAL = True\n", encoding="utf-8")

    from gpu_index import GpuFileIndex

    idx = GpuFileIndex()
    idx.index_directory(str(repo_a))
    assert any(r["file"].endswith("alpha.py") for r in idx.search("ONLY_REPO_A_SIGNAL"))

    # Switch: non-append index should replace corpus entirely
    idx.index_directory(str(repo_b))
    assert not any(r["file"].endswith("alpha.py") for r in idx.search("ONLY_REPO_A_SIGNAL")), \
        "RepoA file appeared in search results after switching to RepoB"
    assert any(r["file"].endswith("beta.py") for r in idx.search("ONLY_REPO_B_SIGNAL")), \
        "RepoB file not found after switching"


# ---------------------------------------------------------------------------
# B. Signal scan root isolation via HTTP
# ---------------------------------------------------------------------------

def test_signal_scan_does_not_return_cross_repo_paths(tmp_path, monkeypatch):
    """When the active index is RepoB, /scan/signals must not return RepoA paths."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()
    repo_a_file = repo_a / "Legacy.cs"
    repo_a_file.write_text("using System.Web;\n", encoding="utf-8")

    repo_b = tmp_path / "RepoB"
    repo_b.mkdir()
    repo_b_file = repo_b / "Modern.cs"
    repo_b_file.write_text("using Microsoft.AspNetCore.Mvc;\n", encoding="utf-8")

    # Active index is RepoB only
    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(repo_b), {str(repo_b_file): repo_b_file.read_text()}),
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())
    monkeypatch.setattr(mcp_server, "_http_roots", [str(repo_b)])

    status, body = _post_http("/scan/signals", {"categories": ["legacy-dotnet"]})
    assert status == 200

    all_abs = [
        m.get("absoluteFile", "")
        for sig in body["signals"]
        for m in sig["matches"]
    ]
    assert not any(str(repo_a) in p for p in all_abs), \
        f"RepoA path leaked into signal scan: {[p for p in all_abs if str(repo_a) in p]}"


# ---------------------------------------------------------------------------
# C. Cache directory exclusion — .gpu-search-cache never in search results
# ---------------------------------------------------------------------------

def test_cache_dir_content_excluded_from_signal_scan(tmp_path, monkeypatch):
    """A .gpu-search-cache file must not appear in /scan/signals results even if injected."""
    repo = tmp_path / "Repo"
    repo.mkdir()
    real_file = repo / "App.cs"
    real_file.write_text("services.AddSingleton<IFoo, Foo>();\n", encoding="utf-8")

    cache_dir = repo / ".gpu-search-cache"
    cache_dir.mkdir()
    cache_json = cache_dir / "files-v1.json"
    # Embed a strong signal keyword inside the cache metadata file
    cache_json.write_text(
        '{"files": [{"path": "fake", "AddSingleton": true}]}\n',
        encoding="utf-8",
    )

    # Inject a fake index that would include the cache file if the filter wasn't there
    contents = {
        str(real_file): real_file.read_text(),
        str(cache_json): cache_json.read_text(),  # should be blocked by boundary filter
    }
    monkeypatch.setattr(mcp_server, "index", _FakeIndex(str(repo), contents))
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())
    monkeypatch.setattr(mcp_server, "_http_roots", [str(repo)])

    status, body = _post_http("/scan/signals", {"categories": ["di"]})
    assert status == 200

    all_abs = [
        m.get("absoluteFile", "")
        for sig in body["signals"]
        for m in sig["matches"]
    ]
    assert not any(".gpu-search-cache" in p for p in all_abs), \
        f".gpu-search-cache path appeared in scan results: {[p for p in all_abs if '.gpu-search-cache' in p]}"


def test_pattern_index_does_not_index_cache_dir(tmp_path):
    """Pattern index discovery must skip .gpu-search-cache directories."""
    repo = tmp_path / "Repo"
    repo.mkdir()
    real_file = repo / "app.py"
    real_file.write_text("ONLY_REAL_CODE = True\n", encoding="utf-8")

    cache_dir = repo / ".gpu-search-cache"
    cache_dir.mkdir()
    cache_json = cache_dir / "files-v1.json"
    cache_json.write_text('{"search_me": "ONLY_REAL_CODE"}\n', encoding="utf-8")

    from gpu_index import GpuFileIndex
    idx = GpuFileIndex()
    idx.index_directory(str(repo))

    for r in idx.search("ONLY_REAL_CODE"):
        assert ".gpu-search-cache" not in r["file"], \
            f".gpu-search-cache path indexed: {r['file']}"


# ---------------------------------------------------------------------------
# D. Cache root mismatch — stale manifest rejected
# ---------------------------------------------------------------------------

def test_pattern_cache_rejected_when_directory_mismatch(tmp_path):
    """Cache with directory=RepoA must be rejected when indexing RepoB."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()
    repo_b = tmp_path / "RepoB"
    repo_b.mkdir()
    (repo_b / "beta.py").write_text("x = 1\n", encoding="utf-8")

    # Write a fake manifest claiming directory=RepoA into RepoB's cache dir
    cache_dir = repo_b / ".gpu-search-cache"
    cache_dir.mkdir()
    manifest = {
        "pattern_version": 1,
        "directory": str(repo_a),   # wrong directory
        "allow_env_files": False,
        "file_count": 0,
        "updated_at": 0.0,
    }
    (cache_dir / "cache-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (cache_dir / "files-v1.json").write_text(
        json.dumps({"files": []}), encoding="utf-8"
    )
    (cache_dir / "pattern-index-v1.bin").write_bytes(b"")

    from gpu_index import GpuFileIndex
    idx = GpuFileIndex()
    # Should rebuild (ignoring the mismatched cache) and find beta.py
    stats = idx.index_directory(str(repo_b))
    assert stats["indexed"] > 0, "Expected RepoB files to be indexed after cache mismatch"


def test_dep_cache_rejected_when_directory_mismatch(tmp_path):
    """Dep cache with directory=RepoA must be rejected when indexing RepoB."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()
    repo_b = tmp_path / "RepoB"
    repo_b.mkdir()
    (repo_b / "mod.py").write_text("import os\n", encoding="utf-8")

    cache_dir = repo_b / ".gpu-search-cache"
    cache_dir.mkdir()
    fake_dep_cache = {
        "version": 2,
        "directory": str(repo_a),  # wrong directory
        "files": [],
        "signatures": {},
        "edges": [],
        "edge_reasons": {},
        "csharp_symbols": {},
        "updated_at": 0.0,
    }
    (cache_dir / "dep-graph-v1.json").write_text(
        json.dumps(fake_dep_cache), encoding="utf-8"
    )

    from gpu_dep_index import DepIndex
    dep = DepIndex()
    stats = dep.index_directory(str(repo_b))
    # Should rebuild — mod.py should be discovered
    assert stats["files"] > 0, "Expected RepoB files in dep index after cache mismatch"


def test_pattern_cache_rejected_with_cache_dir_entries(tmp_path):
    """Pattern cache that lists .gpu-search-cache paths must be rejected."""
    repo = tmp_path / "Repo"
    repo.mkdir()
    (repo / "real.py").write_text("x = 1\n", encoding="utf-8")

    cache_dir = repo / ".gpu-search-cache"
    cache_dir.mkdir()
    # Fake files-v1.json that includes a .gpu-search-cache path
    fake_entries = [
        {"path": str(cache_dir / "files-v1.json"), "size": 10, "mtime_ns": 0,
         "hash": "a" * 32, "offset": 0, "line_offset_start": 0, "line_count": 0},
    ]
    manifest = {
        "pattern_version": 1,
        "directory": str(repo),
        "allow_env_files": False,
        "file_count": 1,
        "updated_at": 0.0,
    }
    (cache_dir / "cache-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (cache_dir / "files-v1.json").write_text(json.dumps({"files": fake_entries}), encoding="utf-8")
    (cache_dir / "pattern-index-v1.bin").write_bytes(b"")

    from gpu_index import GpuFileIndex
    idx = GpuFileIndex()
    # The corrupted cache should be rejected; real.py should be indexed
    stats = idx.index_directory(str(repo))
    assert stats["indexed"] >= 1, "Expected real.py to be indexed after corrupt cache rejected"
    # And the cache file itself must not appear
    for r in idx.search("x"):
        assert ".gpu-search-cache" not in r["file"]


# ---------------------------------------------------------------------------
# E. Diagnostics correctness
# ---------------------------------------------------------------------------

def test_diagnostics_indexed_roots_matches_active_root(tmp_path, monkeypatch):
    """diagnostics_snapshot() indexedRoots should reflect the active index root."""
    repo = tmp_path / "Repo"
    repo.mkdir()
    (repo / "app.py").write_text("x = 1\n", encoding="utf-8")

    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(repo), {str(repo / "app.py"): "x = 1\n"}),
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())
    monkeypatch.setattr(mcp_server, "deps", _FakeDeps())
    monkeypatch.setattr(mcp_server, "_http_roots", [str(repo)])

    snap = mcp_server.diagnostics_snapshot()
    roots = [r["path"] for r in snap["indexedRoots"]]
    assert any(str(repo.resolve()) in r for r in roots), \
        f"Active root {repo} not in indexedRoots: {roots}"


def test_diagnostics_no_stale_root_after_switch(tmp_path, monkeypatch):
    """Diagnostics must not show a root that is no longer active."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()
    repo_b = tmp_path / "RepoB"
    repo_b.mkdir()
    (repo_b / "app.py").write_text("x = 1\n", encoding="utf-8")

    # Active state: only RepoB
    monkeypatch.setattr(
        mcp_server, "index",
        _FakeIndex(str(repo_b), {str(repo_b / "app.py"): "x = 1\n"}),
    )
    monkeypatch.setattr(mcp_server, "semantic", _FakeSemantic())
    monkeypatch.setattr(mcp_server, "deps", _FakeDeps())
    monkeypatch.setattr(mcp_server, "_http_roots", [str(repo_b)])

    snap = mcp_server.diagnostics_snapshot()
    roots = [r["path"] for r in snap["indexedRoots"]]
    assert not any(str(repo_a.resolve()) in r for r in roots), \
        f"Stale RepoA root appeared in indexedRoots: {roots}"


# ---------------------------------------------------------------------------
# G. _prepare_startup — explicit --directory suppresses config dir merging
# ---------------------------------------------------------------------------

def test_prepare_startup_explicit_dir_excludes_config_dirs(tmp_path, monkeypatch):
    """With --directory, all_targets must equal cli_targets only, never config extra dirs."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()
    repo_b = tmp_path / "RepoB"
    repo_b.mkdir()

    # Config has RepoA saved from a previous session
    monkeypatch.setattr(mcp_server, "_load_config_dirs", lambda: [str(repo_a)])
    monkeypatch.setattr(mcp_server, "_save_config_dirs", lambda dirs: None)

    # Simulate starting with --directory RepoB
    args = SimpleNamespace(
        directories=[str(repo_b)],
        allow_env_files=False,
        rebuild_cache=False,
        semantic_model=None,
        device=None,
        download_semantic_model=False,
    )

    # Patch global mutations
    monkeypatch.setattr(mcp_server, "_ALLOW_ENV_FILES", False)

    with patch("mcp_server.resolve_semantic_model_id", return_value="model-id"), \
         patch("mcp_server.set_configured_semantic_model_id"):
        cli_targets, all_targets = mcp_server._prepare_startup(args)

    assert str(repo_a.resolve()) not in [str(Path(t).resolve()) for t in all_targets], \
        f"RepoA (config dir) leaked into all_targets: {all_targets}"
    assert any(str(repo_b.resolve()) in str(Path(t).resolve()) for t in all_targets), \
        f"RepoB not in all_targets: {all_targets}"


def test_prepare_startup_no_explicit_dir_uses_config_dirs(tmp_path, monkeypatch):
    """Without --directory, config extra dirs are included in all_targets (multi-root mode)."""
    repo_a = tmp_path / "RepoA"
    repo_a.mkdir()

    monkeypatch.setattr(mcp_server, "_load_config_dirs", lambda: [str(repo_a)])
    monkeypatch.setattr(mcp_server, "_save_config_dirs", lambda dirs: None)

    args = SimpleNamespace(
        directories=None,  # no --directory
        allow_env_files=False,
        rebuild_cache=False,
        semantic_model=None,
        device=None,
        download_semantic_model=False,
    )
    monkeypatch.setattr(mcp_server, "_ALLOW_ENV_FILES", False)

    monkeypatch.chdir(tmp_path)  # cwd used as default when no --directory

    with patch("mcp_server.resolve_semantic_model_id", return_value="model-id"), \
         patch("mcp_server.set_configured_semantic_model_id"):
        cli_targets, all_targets = mcp_server._prepare_startup(args)

    # RepoA from config should appear in all_targets when no explicit --directory
    all_resolved = [str(Path(t).resolve()) for t in all_targets]
    assert str(repo_a.resolve()) in all_resolved, \
        f"Config dir RepoA not in all_targets in no-explicit-dir mode: {all_targets}"
