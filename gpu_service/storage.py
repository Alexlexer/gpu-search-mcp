"""Replaceable byte transport for packed corpora.

Storage backends know how bytes are obtained.  Search code only supplies an
offset, a length, and a reusable destination.  A future direct-storage backend
can write ``destination.device_buffer`` and return ``device_ready=True``
without changing the search loop or verifier.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import mmap
from pathlib import Path
import threading
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageDestination(Protocol):
    """Destination exposed by the GPU buffer pool to storage transports."""

    @property
    def host_view(self) -> memoryview:
        """Writable byte view used by ordinary host-backed transports."""

    @property
    def device_buffer(self):
        """Framework device allocation available to direct-storage backends."""


@dataclass(frozen=True)
class ReadResult:
    bytes_read: int
    # True means the backend populated device_buffer directly; the pool must
    # not perform a host-to-device copy.
    device_ready: bool = False


def _writable_view(destination: StorageDestination | bytearray | memoryview) -> memoryview:
    view = getattr(destination, "host_view", None)
    if view is None:
        view = memoryview(destination)
    if not isinstance(view, memoryview):
        view = memoryview(view)
    view = view.cast("B")
    if view.readonly:
        raise TypeError("storage destination must be writable")
    return view


class StorageBackend(ABC):
    """Random-access transport over a stable byte address space."""

    @property
    @abstractmethod
    def size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def read(
        self,
        offset: int,
        size: int,
        destination: StorageDestination | bytearray | memoryview,
    ) -> ReadResult:
        """Read at most ``size`` bytes at ``offset`` into ``destination``."""

    @contextmanager
    def read_session(self):
        """Keep transport resources open across a sequence of reads."""
        yield self

    def close(self) -> None:
        """Release backend resources."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class InMemoryStorageBackend(StorageBackend):
    """Random-access backend for tests and explicitly memory-resident corpora."""

    def __init__(self, data: bytes | bytearray | memoryview):
        self._data = memoryview(data).cast("B")

    @property
    def size(self) -> int:
        return len(self._data)

    def read(self, offset: int, size: int, destination) -> ReadResult:
        view = _writable_view(destination)
        count = _bounded_count(offset, size, len(view), self.size)
        if count:
            view[:count] = self._data[offset:offset + count]
        return ReadResult(count)


class FileStorageBackend(StorageBackend):
    """Portable positioned reads without mapping or loading the whole corpus."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None
        self._size = self.path.stat().st_size
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int, size: int, destination) -> ReadResult:
        view = _writable_view(destination)
        count = _bounded_count(offset, size, len(view), self.size)
        total = 0
        # seek/readinto is protected because one backend may later feed several
        # staging buffers concurrently.  A platform pread implementation can be
        # substituted without touching callers.
        with self._lock:
            handle = self._handle
            temporary = handle is None
            if temporary:
                handle = self.path.open("rb", buffering=0)
            try:
                handle.seek(offset)
                while total < count:
                    read = handle.readinto(view[total:count])
                    if not read:
                        break
                    total += read
            finally:
                if temporary:
                    handle.close()
        return ReadResult(total)

    @contextmanager
    def read_session(self):
        with self._lock:
            if self._handle is not None:
                raise RuntimeError("file storage read session is already active")
            self._handle = self.path.open("rb", buffering=0)
        try:
            yield self
        finally:
            self.close()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class MmapStorageBackend(StorageBackend):
    """Memory-mapped backend; pages remain demand-loaded by the operating system."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        self._size = self.path.stat().st_size
        self._mapping = (
            mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
            if self._size
            else None
        )

    @property
    def size(self) -> int:
        return self._size

    def read(self, offset: int, size: int, destination) -> ReadResult:
        view = _writable_view(destination)
        count = _bounded_count(offset, size, len(view), self.size)
        if count and self._mapping is not None:
            view[:count] = self._mapping[offset:offset + count]
        return ReadResult(count)

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._handle.closed:
            self._handle.close()


def _bounded_count(offset: int, size: int, capacity: int, source_size: int) -> int:
    if offset < 0 or size < 0:
        raise ValueError("offset and size must be non-negative")
    if size > capacity:
        raise ValueError(f"destination capacity {capacity} is smaller than read size {size}")
    if offset >= source_size:
        return 0
    return min(size, source_size - offset)
