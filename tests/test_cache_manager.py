"""Tests for explicit persistent cache metadata and invalidation."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server
from cache_manager import (
    CACHE_META_FILENAME,
    CACHE_SCHEMA_VERSION,
    PATTERN_CACHE_SCHEMA_VERSION,
    compute_source_fingerprint,
    load_cache_metadata,
)
from gpu_index import GpuFileIndex, INDEXED_EXTS, SKIP_DIRS


def test_pattern_cache_writes_metadata(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    idx = GpuFileIndex()

    stats = idx.index_directory(str(tmp_path))

    assert stats["cache"] == "rebuilt"
    meta_path = tmp_path / ".gpu-search-cache" / CACHE_META_FILENAME
    assert meta_path.exists()
    metadata = load_cache_metadata(meta_path.parent)
    assert metadata is not None
    assert metadata["schemaVersion"] == CACHE_SCHEMA_VERSION
    entry = next(e for e in metadata["cacheEntries"] if e["name"] == "pattern")
    assert entry["schemaVersion"] == PATTERN_CACHE_SCHEMA_VERSION
    assert entry["status"] == "rebuilt"
    assert entry["sourceFingerprint"]["fileCount"] == 1


def test_current_schema_pattern_cache_is_accepted(tmp_path: Path):
    (tmp_path / "app.py").write_text("needle = True\n", encoding="utf-8")
    first = GpuFileIndex()
    first.index_directory(str(tmp_path))

    second = GpuFileIndex()
    stats = second.index_directory(str(tmp_path))

    assert stats["cache"] == "loaded"
    assert second.search("needle")


def test_missing_metadata_does_not_crash_and_rebuilds(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    idx = GpuFileIndex()
    idx.index_directory(str(tmp_path))
    (tmp_path / ".gpu-search-cache" / CACHE_META_FILENAME).unlink()

    rebuilt = GpuFileIndex().index_directory(str(tmp_path))

    assert rebuilt["indexed"] == 1
    assert rebuilt["cache"] == "rebuilt"
    assert (tmp_path / ".gpu-search-cache" / CACHE_META_FILENAME).exists()


def test_mismatched_schema_invalidates_and_rebuilds(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    idx = GpuFileIndex()
    idx.index_directory(str(tmp_path))
    meta_path = tmp_path / ".gpu-search-cache" / CACHE_META_FILENAME
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["cacheEntries"][0]["schemaVersion"] = 999
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    rebuilt = GpuFileIndex().index_directory(str(tmp_path))

    assert rebuilt["cache"] == "rebuilt"
    metadata = load_cache_metadata(meta_path.parent)
    entry = next(e for e in metadata["cacheEntries"] if e["name"] == "pattern")
    assert entry["schemaVersion"] == PATTERN_CACHE_SCHEMA_VERSION


def test_changed_source_fingerprint_invalidates_and_rebuilds(tmp_path: Path):
    src = tmp_path / "app.py"
    src.write_text("x = 1\n", encoding="utf-8")
    idx = GpuFileIndex()
    idx.index_directory(str(tmp_path))
    before = compute_source_fingerprint(str(tmp_path), INDEXED_EXTS, SKIP_DIRS)

    src.write_text("x = 2\ny = 3\n", encoding="utf-8")
    after = compute_source_fingerprint(str(tmp_path), INDEXED_EXTS, SKIP_DIRS)
    assert before["hash"] != after["hash"]

    rebuilt = GpuFileIndex().index_directory(str(tmp_path))

    assert rebuilt["cache"] == "rebuilt"


def test_rebuild_cache_flag_ignores_valid_pattern_cache(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    idx = GpuFileIndex()
    idx.index_directory(str(tmp_path))

    rebuilt = GpuFileIndex().index_directory(str(tmp_path), force_rebuild=True)

    assert rebuilt["cache"] == "rebuilt"


def test_parse_args_accepts_rebuild_cache_flag():
    args = mcp_server._parse_args(["--rebuild-cache"])

    assert args.rebuild_cache is True


def test_stats_cache_metadata_shape(tmp_path: Path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    GpuFileIndex().index_directory(str(tmp_path))
    old_roots = list(mcp_server._http_roots)
    try:
        mcp_server._http_roots = [str(tmp_path)]
        stats = mcp_server.cache_metadata_for_stats()
    finally:
        mcp_server._http_roots = old_roots

    assert stats["schemaVersion"] == CACHE_SCHEMA_VERSION
    assert stats["directory"].endswith(".gpu-search-cache")
    assert any(e["name"] == "pattern" for e in stats["entries"])
