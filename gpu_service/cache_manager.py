"""Small cache metadata/versioning helpers for persistent gpu-search caches."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 1
PATTERN_CACHE_SCHEMA_VERSION = 1
DEPENDENCY_CACHE_SCHEMA_VERSION = 2
SEMANTIC_CACHE_SCHEMA_VERSION = 1
CACHE_META_FILENAME = "cache-meta.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cache_dir_for_repo(repo_root: str) -> Path:
    return Path(repo_root) / ".gpu-search-cache"


def load_cache_metadata(cache_dir: str | Path) -> dict[str, Any] | None:
    try:
        path = Path(cache_dir) / CACHE_META_FILENAME
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schemaVersion") != CACHE_SCHEMA_VERSION:
            return None
        if not isinstance(data.get("cacheEntries"), list):
            return None
        return data
    except Exception:
        return None


def save_cache_metadata(cache_dir: str | Path, metadata: dict[str, Any]) -> None:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    metadata["updatedAtUtc"] = utc_now()
    (cache_path / CACHE_META_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def new_cache_metadata(repo_root: str, gpu_search_version: str) -> dict[str, Any]:
    now = utc_now()
    return {
        "schemaVersion": CACHE_SCHEMA_VERSION,
        "createdAtUtc": now,
        "updatedAtUtc": now,
        "gpuSearchVersion": gpu_search_version,
        "repoRoot": str(Path(repo_root).resolve()),
        "pythonVersion": sys.version.split()[0],
        "platform": platform.platform(),
        "cacheEntries": [],
    }


def get_cache_entry(metadata: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
    if not metadata:
        return None
    for entry in metadata.get("cacheEntries", []):
        if entry.get("name") == name:
            return entry
    return None


def compute_source_fingerprint(
    repo_root: str,
    include_exts: set[str],
    skip_dirs: set[str],
    max_file_mb: float = 5.0,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo = str(Path(repo_root).resolve())
    max_bytes = int(max_file_mb * 1024 * 1024)
    file_count = 0
    max_mtime_ns = 0
    rows: list[str] = []
    for root, dirs, files in os.walk(repo):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        for fname in sorted(files):
            ext = Path(fname).suffix.lower()
            if not ext and fname.startswith(".") and fname.count(".") == 1:
                ext = fname.lower()
            if ext not in include_exts:
                continue
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
                if st.st_size == 0 or st.st_size > max_bytes:
                    continue
                rel = os.path.relpath(fpath, repo)
                file_count += 1
                max_mtime_ns = max(max_mtime_ns, st.st_mtime_ns)
                rows.append(f"{rel}:{st.st_size}:{st.st_mtime_ns}")
            except OSError:
                continue
    payload = {
        "repoRoot": repo,
        "fileCount": file_count,
        "maxModifiedNs": max_mtime_ns,
        "settings": settings or {},
        "rows": rows,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "hash": digest,
        "repoRoot": repo,
        "fileCount": file_count,
        "maxModifiedNs": max_mtime_ns,
        "settings": settings or {},
    }


def is_cache_entry_valid(
    metadata: dict[str, Any] | None,
    name: str,
    schema_version: int,
    source_fingerprint: dict[str, Any],
) -> bool:
    entry = get_cache_entry(metadata, name)
    if not entry:
        return False
    if entry.get("schemaVersion") != schema_version:
        return False
    stored_fp = entry.get("sourceFingerprint") or {}
    return stored_fp.get("hash") == source_fingerprint.get("hash")


def upsert_cache_entry(
    cache_dir: str | Path,
    repo_root: str,
    gpu_search_version: str,
    *,
    name: str,
    schema_version: int,
    file_path: str | Path,
    source_fingerprint: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    metadata = load_cache_metadata(cache_dir) or new_cache_metadata(repo_root, gpu_search_version)
    now = utc_now()
    entries = [e for e in metadata.get("cacheEntries", []) if e.get("name") != name]
    old = get_cache_entry(metadata, name) or {}
    entries.append({
        "name": name,
        "schemaVersion": schema_version,
        "filePath": str(file_path),
        "createdAtUtc": old.get("createdAtUtc", now),
        "updatedAtUtc": now,
        "sourceFingerprint": source_fingerprint,
        "status": status,
    })
    metadata["cacheEntries"] = sorted(entries, key=lambda e: e.get("name", ""))
    save_cache_metadata(cache_dir, metadata)
    return metadata


def invalidate_cache_entry(cache_dir: str | Path, name: str, status: str = "invalidated") -> None:
    metadata = load_cache_metadata(cache_dir)
    if not metadata:
        return
    changed = False
    for entry in metadata.get("cacheEntries", []):
        if entry.get("name") == name:
            entry["status"] = status
            entry["updatedAtUtc"] = utc_now()
            changed = True
    if changed:
        save_cache_metadata(cache_dir, metadata)
