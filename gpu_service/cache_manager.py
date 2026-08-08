"""Versioned, content-addressed, transactional persistent cache helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

CACHE_SCHEMA_VERSION = 2
PATTERN_CACHE_SCHEMA_VERSION = 2
DEPENDENCY_CACHE_SCHEMA_VERSION = 3
SEMANTIC_CACHE_SCHEMA_VERSION = 2
CACHE_META_FILENAME = "cache-meta.json"
CACHE_LOCK_FILENAME = ".cache.lock"
CACHE_TRANSACTION_FILENAME = ".cache-transaction.json"


class CacheLockTimeout(TimeoutError):
    """Raised when another process holds the repository cache lock."""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cache_dir_for_repo(repo_root: str) -> Path:
    return Path(repo_root) / ".gpu-search-cache"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def cache_content_address(
    name: str,
    schema_version: int,
    source_fingerprint: dict[str, Any],
    gpu_search_version: str,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable identity covering source, schema, app, and producer settings."""
    payload = {
        "name": name,
        "schemaVersion": schema_version,
        "sourceHash": source_fingerprint.get("hash", ""),
        "gpuSearchVersion": gpu_search_version,
        "components": components if components is not None else source_fingerprint.get("settings", {}),
    }
    return {
        "algorithm": "sha256",
        "hash": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        "components": payload["components"],
    }


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_synced_bytes(path: Path, data: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    _atomic_write(Path(path), lambda temporary: _write_synced_bytes(temporary, data))


def atomic_write_json(path: str | Path, data: Any) -> None:
    encoded = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, encoded)


def _safe_transaction_path(cache_dir: Path, value: str) -> Path:
    root = cache_dir.resolve()
    candidate = (root / value).resolve()
    if os.path.commonpath((str(root), str(candidate))) != str(root):
        raise ValueError(f"cache transaction path escapes cache directory: {value}")
    return candidate


def _process_is_running(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def recover_cache_transaction(cache_dir: str | Path) -> bool:
    """Rollback an interrupted commit. Call only while holding the cache lock."""
    root = Path(cache_dir)
    marker_path = root / CACHE_TRANSACTION_FILENAME
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        for item in reversed(marker.get("entries", [])):
            final = _safe_transaction_path(root, item["final"])
            temporary = _safe_transaction_path(root, item["temporary"])
            backup = _safe_transaction_path(root, item["backup"])
            if backup.exists():
                final.unlink(missing_ok=True)
                os.replace(backup, final)
            elif (
                not item.get("hadFinal")
                and marker.get("state") == "committing"
                and not temporary.exists()
            ):
                # Promotion can complete before its phase update reaches the marker.
                final.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)
            backup.unlink(missing_ok=True)
    except Exception:
        # A malformed marker must not make every future cache operation fail.
        pass
    marker_path.unlink(missing_ok=True)
    return True


@contextmanager
def repository_cache_lock(
    cache_dir: str | Path,
    *,
    timeout_seconds: float = 10.0,
    stale_after_seconds: float = 300.0,
) -> Iterator[None]:
    """Acquire a cross-process repository lock with bounded stale-lock recovery."""
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / CACHE_LOCK_FILENAME
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "token": token, "createdAtUtc": utc_now()}, handle)
            break
        except FileExistsError:
            try:
                before = lock_path.stat()
                if time.time() - before.st_mtime > stale_after_seconds:
                    try:
                        owner = json.loads(lock_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        owner = {}
                    after = lock_path.stat()
                    unchanged = (
                        before.st_mtime_ns,
                        before.st_size,
                    ) == (
                        after.st_mtime_ns,
                        after.st_size,
                    )
                    if unchanged and not _process_is_running(owner.get("pid")):
                        lock_path.unlink(missing_ok=True)
                        continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise CacheLockTimeout(f"timed out waiting for cache lock: {lock_path}")
            time.sleep(0.05)
    try:
        recover_cache_transaction(root)
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock_path.unlink(missing_ok=True)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass


class CacheTransaction:
    """Stage related cache artifacts and atomically commit or restore the old set."""

    def __init__(self, cache_dir: str | Path, name: str):
        self.cache_dir = Path(cache_dir)
        self.name = name
        self.id = uuid.uuid4().hex
        self._entries: list[dict[str, Any]] = []
        self._marker = self.cache_dir / CACHE_TRANSACTION_FILENAME

    def _save_marker(self, state: str) -> None:
        atomic_write_json(self._marker, {
            "id": self.id,
            "name": self.name,
            "state": state,
            "pid": os.getpid(),
            "updatedAtUtc": utc_now(),
            "entries": self._entries,
        })

    def stage_writer(self, file_path: str | Path, write: Callable[[Path], None]) -> Path:
        final = Path(file_path).resolve()
        root = self.cache_dir.resolve()
        if os.path.commonpath((str(root), str(final))) != str(root):
            raise ValueError(f"cache artifact escapes cache directory: {final}")
        relative = final.relative_to(root)
        temporary = final.with_name(f".{final.name}.{self.id}.tmp")
        backup = final.with_name(f".{final.name}.{self.id}.bak")
        item = {
            "final": str(relative),
            "temporary": str(temporary.relative_to(root)),
            "backup": str(backup.relative_to(root)),
            "hadFinal": final.exists(),
            "phase": "staged",
        }
        temporary.parent.mkdir(parents=True, exist_ok=True)
        try:
            write(temporary)
            if not temporary.exists():
                raise RuntimeError(f"cache writer did not create {temporary}")
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        self._entries.append(item)
        self._save_marker("staging")
        return final

    def stage_bytes(self, file_path: str | Path, data: bytes) -> Path:
        return self.stage_writer(file_path, lambda temporary: temporary.write_bytes(data))

    def stage_json(self, file_path: str | Path, data: Any) -> Path:
        encoded = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        return self.stage_bytes(file_path, encoded)

    def commit(self) -> None:
        self._save_marker("committing")
        try:
            for item in self._entries:
                final = _safe_transaction_path(self.cache_dir, item["final"])
                temporary = _safe_transaction_path(self.cache_dir, item["temporary"])
                backup = _safe_transaction_path(self.cache_dir, item["backup"])
                if final.exists():
                    os.replace(final, backup)
                item["phase"] = "backed_up"
                self._save_marker("committing")
                os.replace(temporary, final)
                item["phase"] = "promoted"
                self._save_marker("committing")
            for item in self._entries:
                _safe_transaction_path(self.cache_dir, item["backup"]).unlink(missing_ok=True)
            self._marker.unlink(missing_ok=True)
        except Exception:
            recover_cache_transaction(self.cache_dir)
            raise

    def rollback(self) -> None:
        recover_cache_transaction(self.cache_dir)


@contextmanager
def cache_transaction(cache_dir: str | Path, name: str) -> Iterator[CacheTransaction]:
    with repository_cache_lock(cache_dir):
        transaction = CacheTransaction(cache_dir, name)
        try:
            yield transaction
            transaction.commit()
        except Exception:
            transaction.rollback()
            raise


def load_cache_metadata(
    cache_dir: str | Path, *, allow_active_transaction: bool = False
) -> dict[str, Any] | None:
    try:
        cache_path = Path(cache_dir)
        if not allow_active_transaction and (
            cache_path / CACHE_TRANSACTION_FILENAME
        ).exists():
            return None
        path = cache_path / CACHE_META_FILENAME
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
    metadata["updatedAtUtc"] = utc_now()
    with cache_transaction(cache_path, "metadata") as transaction:
        transaction.stage_json(cache_path / CACHE_META_FILENAME, metadata)


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
                content_digest = hashlib.sha256()
                with open(fpath, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        content_digest.update(chunk)
                file_count += 1
                max_mtime_ns = max(max_mtime_ns, st.st_mtime_ns)
                rows.append(f"{rel}:{st.st_size}:{content_digest.hexdigest()}")
            except OSError:
                continue
    payload = {
        "repoRoot": repo,
        "fileCount": file_count,
        "settings": settings or {},
        "rows": rows,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
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
    gpu_search_version: str | None = None,
    components: dict[str, Any] | None = None,
) -> bool:
    entry = get_cache_entry(metadata, name)
    if not entry or entry.get("schemaVersion") != schema_version:
        return False
    stored_fp = entry.get("sourceFingerprint") or {}
    if stored_fp.get("hash") != source_fingerprint.get("hash"):
        return False
    version = gpu_search_version or (metadata or {}).get("gpuSearchVersion", "")
    expected = cache_content_address(name, schema_version, source_fingerprint, version, components)
    return entry.get("contentAddress", {}).get("hash") == expected["hash"]


def _updated_metadata(
    cache_dir: str | Path,
    repo_root: str,
    gpu_search_version: str,
    *,
    name: str,
    schema_version: int,
    file_path: str | Path,
    source_fingerprint: dict[str, Any],
    status: str,
    components: dict[str, Any] | None,
    allow_active_transaction: bool = False,
) -> dict[str, Any]:
    metadata = load_cache_metadata(
        cache_dir, allow_active_transaction=allow_active_transaction
    ) or new_cache_metadata(repo_root, gpu_search_version)
    metadata["gpuSearchVersion"] = gpu_search_version
    now = utc_now()
    entries = [entry for entry in metadata.get("cacheEntries", []) if entry.get("name") != name]
    old = get_cache_entry(metadata, name) or {}
    entries.append({
        "name": name,
        "schemaVersion": schema_version,
        "filePath": str(file_path),
        "createdAtUtc": old.get("createdAtUtc", now),
        "updatedAtUtc": now,
        "sourceFingerprint": source_fingerprint,
        "contentAddress": cache_content_address(
            name, schema_version, source_fingerprint, gpu_search_version, components
        ),
        "status": status,
    })
    metadata["cacheEntries"] = sorted(entries, key=lambda entry: entry.get("name", ""))
    metadata["updatedAtUtc"] = now
    return metadata


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
    components: dict[str, Any] | None = None,
    transaction: CacheTransaction | None = None,
) -> dict[str, Any]:
    if transaction is not None:
        metadata = _updated_metadata(
            cache_dir, repo_root, gpu_search_version, name=name,
            schema_version=schema_version, file_path=file_path,
            source_fingerprint=source_fingerprint, status=status, components=components,
            allow_active_transaction=True,
        )
        transaction.stage_json(Path(cache_dir) / CACHE_META_FILENAME, metadata)
        return metadata
    with cache_transaction(cache_dir, f"metadata:{name}") as own_transaction:
        metadata = _updated_metadata(
            cache_dir, repo_root, gpu_search_version, name=name,
            schema_version=schema_version, file_path=file_path,
            source_fingerprint=source_fingerprint, status=status, components=components,
            allow_active_transaction=True,
        )
        own_transaction.stage_json(Path(cache_dir) / CACHE_META_FILENAME, metadata)
        return metadata


def invalidate_cache_entry(cache_dir: str | Path, name: str, status: str = "invalidated") -> None:
    with repository_cache_lock(cache_dir):
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
            metadata["updatedAtUtc"] = utc_now()
            atomic_write_json(Path(cache_dir) / CACHE_META_FILENAME, metadata)
