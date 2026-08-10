"""Out-of-core exact search over a versioned packed repository corpus."""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np
import torch

from cache_manager import (
    PATTERN_CACHE_SCHEMA_VERSION,
    compute_source_fingerprint,
    invalidate_cache_entry,
    is_cache_entry_valid,
    load_cache_metadata,
    repository_cache_lock,
    upsert_cache_entry,
)
from candidates import (
    AllChunksCandidateSelector,
    CandidateBuildStats,
    CandidateSelector,
    TrigramCandidateSelector,
    resolve_candidates,
)
from device import DeviceInfo, resolve_torch_device
from gpu_buffer import (
    GpuBuffer,
    GpuBufferPool,
    ReadStats,
    TransferStats,
    _synchronize,
)
from gpu_search import TorchByteSearch
from packed_corpus import (
    DEFAULT_CHUNK_SIZE,
    FORMAT_VERSION as PACKED_FORMAT_VERSION,
    PACKED_DIRNAME,
    BuildStats,
    CorpusChunk,
    PackedCorpusCatalog,
    build_packed_corpus,
)
from server_config import VERSION
from storage import (
    FileStorageBackend,
    InMemoryStorageBackend,
    MmapStorageBackend,
    StorageBackend,
)


DEVICE_INFO: DeviceInfo = resolve_torch_device(os.environ.get("GPU_SEARCH_DEVICE"))
DEVICE = torch.device(DEVICE_INFO.torch_device)


def _best_device() -> torch.device:
    return DEVICE


def _file_ext(fname: str) -> str:
    lower = fname.lower()
    if lower == ".env" or lower.startswith(".env."):
        return ".env"
    return Path(lower).suffix


INDEXED_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".cpp", ".h",
    ".hpp", ".java", ".cs", ".rb", ".php", ".swift", ".kt", ".json", ".yaml",
    ".yml", ".toml", ".md", ".txt", ".html", ".css", ".scss", ".sql", ".sh",
    ".bat", ".ps1", ".cfg", ".ini", ".xml",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    ".next", ".nuxt", "target", "bin", "obj", ".idea", ".vscode", ".mypy_cache",
    ".gpu-search-cache", PACKED_DIRNAME,
}


def _pattern_cache_components(allow_env_files: bool, chunk_size: int) -> dict:
    return {
        "parser": "byte-pattern-out-of-core-v1",
        "lineOffsets": "files-index-json-v1",
        "packedCorpus": PACKED_FORMAT_VERSION,
        "chunkSize": chunk_size,
        "allowEnvFiles": allow_env_files,
    }


@dataclass
class QueryMetrics:
    total_corpus_size: int = 0
    chunk_size: int = 0
    number_of_chunks: int = 0
    candidate_chunks: int = 0
    candidate_percentage: float = 0.0
    bytes_read_from_storage: int = 0
    bytes_transferred_to_gpu: int = 0
    device_ready_bytes: int = 0
    host_to_gpu_bytes: int = 0
    corpus_percentage_physically_read: float = 0.0
    storage_read_seconds: float = 0.0
    host_to_gpu_seconds: float = 0.0
    gpu_search_seconds: float = 0.0
    total_query_seconds: float = 0.0
    vram_bytes: int = 0
    pipeline_enabled: bool = False
    prefetched_chunks: int = 0


StorageFactory = Callable[[Path], StorageBackend]


def _in_memory_storage(path: Path) -> StorageBackend:
    """Explicit benchmark/test backend; unlike normal backends, loads all bytes."""
    return InMemoryStorageBackend(path.read_bytes())


class GpuFileIndex:
    """Packed, chunked exact-search index with a replaceable byte transport."""

    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        buffer_count: int = 2,
        storage_backend: str | StorageFactory = "file",
        candidate_selector: CandidateSelector | str | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if buffer_count <= 0:
            raise ValueError("buffer_count must be positive")
        self.chunk_size = chunk_size
        self.buffer_count = buffer_count
        self._storage_factory = self._resolve_storage_factory(storage_backend)
        self._candidate_selector = self._resolve_candidate_selector(candidate_selector)
        self._verifier = TorchByteSearch(DEVICE)
        self._read_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gpusearch-read"
        )
        self._pool: Optional[GpuBufferPool] = None
        self._storage: Optional[StorageBackend] = None
        self._catalog: Optional[PackedCorpusCatalog] = None
        self._file_names: list[str] = []
        self._file_meta: dict[str, dict] = {}
        self._cache_status = "cold"
        self._last_query_metrics = QueryMetrics()
        self._last_build_stats: Optional[BuildStats] = None
        self._candidate_build_stats = CandidateBuildStats()
        self.base_dir: Optional[str] = None
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_candidate_selector(
        value: CandidateSelector | str | None,
    ) -> CandidateSelector:
        if value is None or value == "all":
            return AllChunksCandidateSelector()
        if value == "trigram":
            return TrigramCandidateSelector()
        if isinstance(value, CandidateSelector):
            return value
        raise ValueError(
            "candidate_selector must be 'all', 'trigram', or a CandidateSelector"
        )

    @staticmethod
    def _resolve_storage_factory(value: str | StorageFactory) -> StorageFactory:
        if callable(value):
            return value
        if value == "file":
            return FileStorageBackend
        if value == "mmap":
            return MmapStorageBackend
        if value == "memory":
            return _in_memory_storage
        raise ValueError(
            "storage_backend must be 'file', 'mmap', 'memory', or a factory"
        )

    def _cache_dir(self, directory: str) -> Path:
        return Path(directory) / ".gpu-search-cache"

    def _packed_dir(self, directory: str) -> Path:
        return Path(directory) / PACKED_DIRNAME

    def _signature(self, fpath: str) -> Optional[dict]:
        try:
            stat = os.stat(fpath)
            return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        except OSError:
            return None

    def _discover_files(
        self, directory: str, max_bytes: int, effective_exts: set[str]
    ) -> tuple[list[str], int]:
        files: list[str] = []
        skipped = 0
        for root, dirs, names in os.walk(directory):
            dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
            for name in names:
                if _file_ext(name) not in effective_exts:
                    skipped += 1
                    continue
                path = os.path.join(root, name)
                try:
                    if os.path.getsize(path) > max_bytes:
                        skipped += 1
                        continue
                    files.append(os.path.abspath(path))
                except OSError:
                    skipped += 1
        files.sort()
        return files, skipped

    def _load_catalog(self, directory: str, discovered: list[str]) -> PackedCorpusCatalog | None:
        try:
            catalog = PackedCorpusCatalog.load(self._packed_dir(directory))
            if catalog.root != Path(directory).resolve() or catalog.chunk_size != self.chunk_size:
                return None
            if [catalog.absolute_path(entry) for entry in catalog.files] != discovered:
                return None
            return catalog
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _install_catalog(self, catalog: PackedCorpusCatalog) -> None:
        if self._storage is not None:
            self._storage.close()
        self._storage = self._storage_factory(catalog.corpus_path)
        self._catalog = catalog
        self._file_names = [catalog.absolute_path(entry) for entry in catalog.files]
        self._file_meta = {
            path: {
                "file_id": entry.file_id,
                "relative_path": entry.relative_path,
                "offset": entry.offset,
                "length": entry.length,
                "size": entry.size,
                "mtime_ns": entry.mtime_ns,
                "hash": entry.digest,
            }
            for path, entry in zip(self._file_names, catalog.files)
        }
        if self._pool is None:
            self._pool = GpuBufferPool(self.chunk_size, self.buffer_count, DEVICE)

    def index_directory(
        self,
        directory: str,
        max_file_mb: float = 5.0,
        append: bool = False,
        allow_env_files: bool = False,
        force_rebuild: bool = False,
    ) -> dict:
        directory = os.path.abspath(directory)
        max_bytes = int(max_file_mb * 1024 * 1024)
        effective_exts = INDEXED_EXTS | ({".env"} if allow_env_files else set())
        discovered, skipped = self._discover_files(directory, max_bytes, effective_exts)
        fingerprint = compute_source_fingerprint(
            directory,
            effective_exts,
            SKIP_DIRS,
            max_file_mb=max_file_mb,
            settings={
                "allow_env_files": allow_env_files,
                "cache": "pattern",
                "chunk_size": self.chunk_size,
                "packed_version": PACKED_FORMAT_VERSION,
            },
        )
        metadata = load_cache_metadata(self._cache_dir(directory))
        components = _pattern_cache_components(allow_env_files, self.chunk_size)
        entry_valid = is_cache_entry_valid(
            metadata,
            "pattern",
            PATTERN_CACHE_SCHEMA_VERSION,
            fingerprint,
            VERSION,
            components,
        )
        if force_rebuild:
            invalidate_cache_entry(self._cache_dir(directory), "pattern", "rebuild_requested")
        elif metadata is not None and not entry_valid:
            invalidate_cache_entry(self._cache_dir(directory), "pattern", "stale")

        with self._lock:
            existing: list[str] = []
            if append and self._file_names:
                current_root = Path(directory).resolve()
                for name in self._file_names:
                    try:
                        Path(name).resolve().relative_to(current_root)
                    except ValueError:
                        if Path(name).exists():
                            existing.append(name)
            else:
                self.base_dir = directory

            catalog = None
            cache_status = "rebuilt"
            build_stats = None
            if not append and not force_rebuild and entry_valid:
                catalog = self._load_catalog(directory, discovered)
                if catalog is not None:
                    cache_status = "loaded"

            if catalog is None:
                if self._storage is not None:
                    self._storage.close()
                    self._storage = None
                root = self.base_dir or directory
                packed_dir = self._packed_dir(root)
                # Several Codex/Claude sessions can start the same repository
                # concurrently. Serialize the packed artifact set separately
                # from cache metadata transactions.
                with repository_cache_lock(
                    packed_dir, timeout_seconds=600.0
                ):
                    if not append and not force_rebuild:
                        refreshed_metadata = load_cache_metadata(
                            self._cache_dir(directory)
                        )
                        refreshed_valid = is_cache_entry_valid(
                            refreshed_metadata,
                            "pattern",
                            PATTERN_CACHE_SCHEMA_VERSION,
                            fingerprint,
                            VERSION,
                            components,
                        )
                        if refreshed_valid:
                            catalog = self._load_catalog(
                                directory, discovered
                            )
                            if catalog is not None:
                                cache_status = "loaded"
                    if catalog is None:
                        catalog, build_stats = build_packed_corpus(
                            root,
                            existing + discovered,
                            packed_dir=packed_dir,
                            chunk_size=self.chunk_size,
                        )
                        self._last_build_stats = build_stats
                    if not append and cache_status != "loaded":
                        upsert_cache_entry(
                            self._cache_dir(directory),
                            directory,
                            VERSION,
                            name="pattern",
                            schema_version=PATTERN_CACHE_SCHEMA_VERSION,
                            file_path=catalog.corpus_path,
                            source_fingerprint=fingerprint,
                            status=cache_status,
                            components=components,
                        )
            self._install_catalog(catalog)
            if self._storage is None:
                raise RuntimeError("packed corpus storage was not initialized")
            with self._storage.read_session():
                self._candidate_build_stats = self._candidate_selector.prepare(
                    catalog, self._storage
                )
            self._cache_status = cache_status

        return {
            "indexed": len(discovered),
            "skipped": skipped,
            "vram_mb": round(self._vram_bytes() / 1024 / 1024, 2),
            "cache": self._cache_status,
            "corpus_bytes": catalog.corpus_size,
            "chunks": len(catalog.chunks),
            "chunk_size": catalog.chunk_size,
            "build_seconds": build_stats.build_time_seconds if build_stats else 0.0,
            "candidate_selector": type(self._candidate_selector).__name__,
            "candidate_index_bytes_read": self._candidate_build_stats.bytes_read,
            "candidate_index_build_seconds": self._candidate_build_stats.build_seconds,
            "candidate_index_load_seconds": self._candidate_build_stats.load_seconds,
            "candidate_index_keys": self._candidate_build_stats.indexed_keys,
            "candidate_index_cache": self._candidate_build_stats.cache_status,
            "candidate_index_bytes": self._candidate_build_stats.index_bytes,
        }

    def update_file(self, fpath: str, allow_env_files: bool = False):
        """Repack after a change; normal queries never open original source files."""
        fpath = os.path.abspath(fpath)
        effective_exts = INDEXED_EXTS | ({".env"} if allow_env_files else set())
        if _file_ext(Path(fpath).name) not in effective_exts or not self.base_dir:
            return
        self.index_directory(
            self.base_dir,
            allow_env_files=allow_env_files,
            force_rebuild=True,
        )
        self._cache_status = "updated"

    def clear(self) -> None:
        """Release the active catalog, storage, and GPU buffers."""
        with self._lock:
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            if self._pool is not None:
                self._pool.close()
                self._pool = None
            self._catalog = None
            self._file_names = []
            self._file_meta = {}
            self.base_dir = None
            self._cache_status = "cold"
            self._last_build_stats = None
            self._candidate_build_stats = CandidateBuildStats()
            self._last_query_metrics = QueryMetrics()

    def search(
        self, pattern: str, case_sensitive: bool = False, max_files: int = 50
    ) -> list[dict]:
        with self._lock:
            if self._storage is None:
                return self._search_locked(pattern, case_sensitive, max_files)
            with self._storage.read_session():
                return self._search_locked(pattern, case_sensitive, max_files)

    def _candidate_buffers(
        self,
        candidates: list[CorpusChunk],
        query_length: int,
        catalog: PackedCorpusCatalog,
        storage: StorageBackend,
        pool: GpuBufferPool,
    ) -> Iterator[tuple[CorpusChunk, GpuBuffer, int, TransferStats]]:
        def read_length(chunk: CorpusChunk) -> int:
            return min(
                chunk.valid_length + query_length - 1,
                catalog.corpus_size - chunk.offset,
            )

        if pool.count < 2 or len(candidates) < 2:
            for chunk in candidates:
                size = read_length(chunk)
                with pool.acquire() as buffer:
                    yield chunk, buffer, size, buffer.load(
                        storage, chunk.offset, size
                    )
            return

        def schedule(
            chunk: CorpusChunk,
        ) -> tuple[CorpusChunk, GpuBuffer, int, Future[ReadStats]]:
            size = read_length(chunk)
            buffer = pool.acquire_buffer()
            try:
                future = self._read_executor.submit(
                    buffer.read_from, storage, chunk.offset, size
                )
            except BaseException:
                pool.release_buffer(buffer)
                raise
            return chunk, buffer, size, future

        pending = schedule(candidates[0])
        try:
            for index in range(len(candidates)):
                current = pending
                pending = None
                chunk, buffer, size, future = current
                try:
                    read = future.result()
                    if index + 1 < len(candidates):
                        pending = schedule(candidates[index + 1])
                    transfer = buffer.make_device_ready(size, read)
                    yield chunk, buffer, size, transfer
                finally:
                    pool.release_buffer(buffer)
        finally:
            if pending is not None:
                _, buffer, _, future = pending
                future.cancel()
                try:
                    future.result()
                except BaseException:
                    pass
                pool.release_buffer(buffer)

    def _search_locked(
        self, pattern: str, case_sensitive: bool = False, max_files: int = 50
    ) -> list[dict]:
        started = time.perf_counter()
        catalog = self._catalog
        storage = self._storage
        pool = self._pool
        if not pattern or catalog is None or storage is None or pool is None:
            self._last_query_metrics = QueryMetrics(
                total_query_seconds=time.perf_counter() - started
            )
            return []

        query = self._verifier.prepare(pattern, case_sensitive)
        if query.length == 0:
            return []
        candidates = resolve_candidates(self._candidate_selector, query.encoded, catalog)
        pipeline_enabled = pool.count >= 2 and len(candidates) >= 2
        metrics = QueryMetrics(
            total_corpus_size=catalog.corpus_size,
            chunk_size=catalog.chunk_size,
            number_of_chunks=len(catalog.chunks),
            candidate_chunks=len(candidates),
            candidate_percentage=(
                100.0 * len(candidates) / len(catalog.chunks)
                if catalog.chunks else 0.0
            ),
            pipeline_enabled=pipeline_enabled,
            prefetched_chunks=max(0, len(candidates) - 1) if pipeline_enabled else 0,
        )
        pool.ensure_capacity(catalog.chunk_size + max(0, query.length - 1))
        hits: dict[int, list[int]] = defaultdict(list)
        hit_lines: dict[int, set[int]] = defaultdict(set)
        matched_file_ids: set[int] = set()
        file_starts = np.fromiter(
            (entry.offset for entry in catalog.files), dtype=np.int64
        )
        file_ends = np.fromiter(
            (entry.offset + entry.length for entry in catalog.files), dtype=np.int64
        )

        for chunk, buffer, read_length, transfer in self._candidate_buffers(
            candidates, query.length, catalog, storage, pool
        ):
            metrics.bytes_read_from_storage += transfer.bytes_read
            metrics.storage_read_seconds += transfer.read_seconds
            metrics.bytes_transferred_to_gpu += read_length
            if transfer.device_ready:
                metrics.device_ready_bytes += read_length
            metrics.host_to_gpu_bytes += transfer.host_to_device_bytes
            metrics.host_to_gpu_seconds += transfer.host_to_device_seconds
            kernel_started = time.perf_counter()
            local_matches = self._verifier.search(
                buffer.device_buffer, read_length, query
            ).cpu().numpy()
            _synchronize(DEVICE)
            metrics.gpu_search_seconds += time.perf_counter() - kernel_started

            owned = local_matches[local_matches < chunk.valid_length]
            if not owned.size or not file_starts.size:
                continue
            corpus_offsets = owned + chunk.offset
            file_ids = np.searchsorted(file_starts, corpus_offsets, side="right") - 1
            in_catalog = (file_ids >= 0) & (file_ids < len(file_starts))
            corpus_offsets = corpus_offsets[in_catalog]
            file_ids = file_ids[in_catalog]
            if not file_ids.size:
                continue
            in_file = corpus_offsets + query.length <= file_ends[file_ids]
            corpus_offsets = corpus_offsets[in_file]
            file_ids = file_ids[in_file]
            if not file_ids.size:
                continue

            unique_file_ids = np.unique(file_ids)
            matched_file_ids.update(int(file_id) for file_id in unique_file_ids)
            for raw_file_id in unique_file_ids:
                file_id = int(raw_file_id)
                if len(hit_lines[file_id]) >= 10:
                    continue
                entry = catalog.file_by_id(file_id)
                file_offsets = corpus_offsets[file_ids == raw_file_id] - entry.offset
                newlines = np.asarray(entry.newline_offsets, dtype=np.int64)
                lines = np.searchsorted(newlines, file_offsets, side="right") + 1
                _, first_indices = np.unique(lines, return_index=True)
                first_indices.sort()
                for first_index in first_indices:
                    line = int(lines[first_index])
                    if line in hit_lines[file_id]:
                        continue
                    hit_lines[file_id].add(line)
                    hits[file_id].append(int(file_offsets[first_index]))
                    if len(hit_lines[file_id]) >= 10:
                        break

        total_match_files = len(matched_file_ids)
        results: list[dict] = []
        for file_id in sorted(hits)[:max(0, max_files)]:
            entry = catalog.file_by_id(file_id)
            matches: list[dict] = []
            seen_lines: set[int] = set()
            for file_offset in hits[file_id]:
                line, line_start, line_end = catalog.line_for_offset(entry, file_offset)
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                content, read_seconds = self._read_bytes(
                    storage, entry.offset + line_start, line_end - line_start
                )
                metrics.bytes_read_from_storage += len(content)
                metrics.storage_read_seconds += read_seconds
                matches.append({
                    "line": line,
                    "content": content.decode("utf-8", errors="replace").rstrip(),
                })
                if len(matches) >= 10:
                    break
            results.append({
                "file": catalog.absolute_path(entry),
                "matches": matches,
                "_total_files": total_match_files,
            })

        metrics.corpus_percentage_physically_read = (
            100.0 * metrics.bytes_read_from_storage / catalog.corpus_size
            if catalog.corpus_size else 0.0
        )
        metrics.vram_bytes = self._vram_bytes()
        metrics.total_query_seconds = time.perf_counter() - started
        self._last_query_metrics = metrics
        return results

    @staticmethod
    def _read_bytes(
        storage: StorageBackend, offset: int, size: int
    ) -> tuple[bytes, float]:
        destination = bytearray(size)
        started = time.perf_counter()
        result = storage.read(offset, size, destination)
        elapsed = time.perf_counter() - started
        if result.bytes_read != size:
            raise EOFError(
                f"short result read at {offset}: expected {size}, got {result.bytes_read}"
            )
        return bytes(destination), elapsed

    def _vram_bytes(self) -> int:
        return self._pool.allocated_device_bytes if self._pool is not None else 0

    def stats(self) -> dict:
        catalog = self._catalog
        result = {
            "files": len(self._file_names),
            "vram_mb": round(self._vram_bytes() / 1024 / 1024, 2),
            "base_dir": self.base_dir,
            "cache": self._cache_status,
            "storage_backend": type(self._storage).__name__ if self._storage else None,
            "corpus_bytes": catalog.corpus_size if catalog else 0,
            "chunk_size": catalog.chunk_size if catalog else self.chunk_size,
            "chunks": len(catalog.chunks) if catalog else 0,
            "buffer_count": self.buffer_count,
            "candidate_selector": type(self._candidate_selector).__name__,
            "candidate_index": asdict(self._candidate_build_stats),
            "last_query": asdict(self._last_query_metrics),
            "last_build": (
                asdict(self._last_build_stats) if self._last_build_stats else None
            ),
        }
        if DEVICE.type == "cuda":
            result["vram_total_mb"] = round(
                torch.cuda.get_device_properties(DEVICE).total_memory / 1024 / 1024
            )
            result["vram_reserved_mb"] = round(
                torch.cuda.memory_reserved(DEVICE) / 1024 / 1024, 2
            )
        elif DEVICE.type == "mps":
            try:
                result["vram_allocated_mb"] = round(
                    torch.mps.current_allocated_memory() / 1024 / 1024, 2
                )
            except Exception:
                pass
        return result

    def close(self) -> None:
        with self._lock:
            if self._storage is not None:
                self._storage.close()
                self._storage = None
            if self._pool is not None:
                self._pool.close()
                self._pool = None
            self._read_executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        try:
            if self._storage is not None:
                self._storage.close()
            self._read_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
