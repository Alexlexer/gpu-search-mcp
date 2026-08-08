"""Tests for explicit persistent cache metadata and invalidation."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import cache_manager
import mcp_server
from cache_manager import (
    CACHE_META_FILENAME,
    CACHE_SCHEMA_VERSION,
    CACHE_TRANSACTION_FILENAME,
    PATTERN_CACHE_SCHEMA_VERSION,
    cache_content_address,
    cache_transaction,
    compute_source_fingerprint,
    load_cache_metadata,
    repository_cache_lock,
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


def test_content_address_covers_source_schema_app_and_components():
    source = {"hash": "source-a", "settings": {"cache": "pattern"}}
    baseline = cache_content_address(
        "pattern", PATTERN_CACHE_SCHEMA_VERSION, source, "1.0", {"parser": "v1"}
    )

    assert baseline == cache_content_address(
        "pattern", PATTERN_CACHE_SCHEMA_VERSION, source, "1.0", {"parser": "v1"}
    )
    assert baseline["hash"] != cache_content_address(
        "pattern", PATTERN_CACHE_SCHEMA_VERSION, source, "2.0", {"parser": "v1"}
    )["hash"]
    assert baseline["hash"] != cache_content_address(
        "pattern", PATTERN_CACHE_SCHEMA_VERSION, source, "1.0", {"parser": "v2"}
    )["hash"]
    assert baseline["hash"] != cache_content_address(
        "pattern", PATTERN_CACHE_SCHEMA_VERSION, {"hash": "source-b"}, "1.0", {"parser": "v1"}
    )["hash"]


def test_cache_transaction_rolls_back_staged_artifacts(tmp_path: Path):
    cache_dir = tmp_path / ".gpu-search-cache"
    artifact = cache_dir / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"old")

    try:
        with cache_transaction(cache_dir, "test") as transaction:
            transaction.stage_bytes(artifact, b"new")
            transaction.stage_json(cache_dir / "new.json", {"value": 1})
            raise RuntimeError("simulated writer failure")
    except RuntimeError:
        pass

    assert artifact.read_bytes() == b"old"
    assert not (cache_dir / "new.json").exists()
    assert not (cache_dir / CACHE_TRANSACTION_FILENAME).exists()
    assert not list(cache_dir.glob("*.tmp"))
    assert not list(cache_dir.glob("*.bak"))


def test_stale_lock_is_recovered(tmp_path: Path):
    cache_dir = tmp_path / ".gpu-search-cache"
    cache_dir.mkdir()
    lock_path = cache_dir / ".cache.lock"
    lock_path.write_text('{"pid": -1, "token": "stale"}', encoding="utf-8")

    with repository_cache_lock(
        cache_dir, timeout_seconds=0.2, stale_after_seconds=-1
    ):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_pattern_cache_commits_artifacts_and_content_address(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")

    GpuFileIndex().index_directory(str(tmp_path))

    cache_dir = tmp_path / ".gpu-search-cache"
    metadata = load_cache_metadata(cache_dir)
    entry = next(item for item in metadata["cacheEntries"] if item["name"] == "pattern")
    assert entry["contentAddress"]["algorithm"] == "sha256"
    assert len(entry["contentAddress"]["hash"]) == 64
    assert not (cache_dir / CACHE_TRANSACTION_FILENAME).exists()
    assert not (cache_dir / ".cache.lock").exists()

def test_source_fingerprint_uses_content_not_mtime(tmp_path: Path):
    source = tmp_path / "app.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = compute_source_fingerprint(str(tmp_path), {".py"}, set())

    source.touch()
    touched = compute_source_fingerprint(str(tmp_path), {".py"}, set())
    source.write_text("value = 2\n", encoding="utf-8")
    changed = compute_source_fingerprint(str(tmp_path), {".py"}, set())

    assert touched["hash"] == first["hash"]
    assert changed["hash"] != first["hash"]


def test_interrupted_commit_is_detected_and_recovered(tmp_path: Path):
    cache_dir = tmp_path / ".gpu-search-cache"
    cache_dir.mkdir()
    artifact = cache_dir / "artifact.bin"
    backup = cache_dir / ".artifact.bin.tx.bak"
    temporary = cache_dir / ".artifact.bin.tx.tmp"
    artifact.write_bytes(b"partial-new")
    backup.write_bytes(b"old")
    marker = {
        "state": "committing",
        "entries": [{
            "final": "artifact.bin",
            "temporary": temporary.name,
            "backup": backup.name,
            "hadFinal": True,
            "phase": "promoted",
        }],
    }
    (cache_dir / CACHE_TRANSACTION_FILENAME).write_text(
        json.dumps(marker), encoding="utf-8"
    )

    assert load_cache_metadata(cache_dir) is None
    with repository_cache_lock(cache_dir):
        pass

    assert artifact.read_bytes() == b"old"
    assert not backup.exists()
    assert not (cache_dir / CACHE_TRANSACTION_FILENAME).exists()


def test_commit_failure_restores_all_previous_artifacts(
    tmp_path: Path, monkeypatch
):
    cache_dir = tmp_path / ".gpu-search-cache"
    cache_dir.mkdir()
    first = cache_dir / "first.bin"
    second = cache_dir / "second.bin"
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    real_replace = cache_manager.os.replace

    def fail_second_promotion(source, destination):
        if Path(destination) == second.resolve() and str(source).endswith(".tmp"):
            raise OSError("simulated promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(cache_manager.os, "replace", fail_second_promotion)
    try:
        with cache_transaction(cache_dir, "failure") as transaction:
            transaction.stage_bytes(first, b"new-first")
            transaction.stage_bytes(second, b"new-second")
    except OSError:
        pass

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"
    assert not (cache_dir / CACHE_TRANSACTION_FILENAME).exists()

def test_recovery_removes_new_artifact_promoted_before_phase_update(
    tmp_path: Path,
):
    cache_dir = tmp_path / ".gpu-search-cache"
    cache_dir.mkdir()
    artifact = cache_dir / "new.bin"
    temporary = cache_dir / ".new.bin.tx.tmp"
    artifact.write_bytes(b"partial")
    marker = {
        "state": "committing",
        "entries": [{
            "final": artifact.name,
            "temporary": temporary.name,
            "backup": ".new.bin.tx.bak",
            "hadFinal": False,
            "phase": "backed_up",
        }],
    }
    (cache_dir / CACHE_TRANSACTION_FILENAME).write_text(
        json.dumps(marker), encoding="utf-8"
    )

    with repository_cache_lock(cache_dir):
        pass

    assert not artifact.exists()
    assert not (cache_dir / CACHE_TRANSACTION_FILENAME).exists()
