"""Reusable staging/device buffers for out-of-core verification."""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import threading
import time
from typing import Iterator

import torch

from storage import ReadResult, StorageBackend


@dataclass(frozen=True)
class ReadStats:
    bytes_read: int
    read_seconds: float
    device_ready: bool


@dataclass(frozen=True)
class TransferStats:
    bytes_read: int
    read_seconds: float
    host_to_device_bytes: int
    host_to_device_seconds: float
    device_ready: bool


class GpuBuffer:
    """One reusable host staging allocation plus one device allocation."""

    def __init__(self, capacity: int, device: torch.device):
        if capacity <= 0:
            raise ValueError("buffer capacity must be positive")
        self.capacity = capacity
        self.device = device
        pin = device.type == "cuda"
        self._host = torch.empty(capacity, dtype=torch.uint8, pin_memory=pin)
        self._device = self._host if device.type == "cpu" else torch.empty(
            capacity, dtype=torch.uint8, device=device
        )

    @property
    def host_view(self) -> memoryview:
        return memoryview(self._host.numpy()).cast("B")

    @property
    def device_buffer(self) -> torch.Tensor:
        return self._device

    @property
    def allocated_device_bytes(self) -> int:
        return self._device.nbytes if self.device.type != "cpu" else 0

    def read_from(
        self, backend: StorageBackend, offset: int, size: int
    ) -> ReadStats:
        if size > self.capacity:
            raise ValueError(f"read size {size} exceeds buffer capacity {self.capacity}")
        read_started = time.perf_counter()
        result: ReadResult = backend.read(offset, size, self)
        read_seconds = time.perf_counter() - read_started
        if result.bytes_read != size:
            raise EOFError(
                f"short corpus read at {offset}: expected {size}, "
                f"got {result.bytes_read}"
            )
        return ReadStats(
            bytes_read=result.bytes_read,
            read_seconds=read_seconds,
            device_ready=result.device_ready,
        )

    def make_device_ready(self, size: int, read: ReadStats) -> TransferStats:
        if size > self.capacity:
            raise ValueError(f"transfer size {size} exceeds buffer capacity {self.capacity}")
        copied = 0
        transfer_seconds = 0.0
        if not read.device_ready and self._device is not self._host:
            transfer_started = time.perf_counter()
            self._device[:size].copy_(
                self._host[:size], non_blocking=self.device.type == "cuda"
            )
            _synchronize(self.device)
            transfer_seconds = time.perf_counter() - transfer_started
            copied = size
        return TransferStats(
            bytes_read=read.bytes_read,
            read_seconds=read.read_seconds,
            host_to_device_bytes=copied,
            host_to_device_seconds=transfer_seconds,
            device_ready=read.device_ready,
        )

    def load(self, backend: StorageBackend, offset: int, size: int) -> TransferStats:
        """Backward-compatible synchronous composition of read and transfer."""
        return self.make_device_ready(size, self.read_from(backend, offset, size))



class GpuBufferPool:
    """Fixed-count pool with acquire/release semantics and grow-on-query capacity."""

    def __init__(self, buffer_size: int, count: int, device: torch.device,
                 *, max_owned_bytes: int = 64 * 1024 * 1024):
        if count <= 0 or buffer_size <= 0 or max_owned_bytes <= 0:
            raise ValueError("buffer size, count and budget must be positive")
        self.buffer_size = buffer_size
        self.count = count
        self.device = device
        self.max_owned_bytes = max_owned_bytes
        # CPU aliases staging/device memory; accelerators require both allocations.
        self._copies = 1 if getattr(device, 'type', 'cpu') == 'cpu' else 2
        self._check_budget(buffer_size)
        self._condition = threading.Condition()
        self._closed = False
        self._leased = 0
        self._leases: dict[int, GpuBuffer] = {}
        self._available: deque[GpuBuffer] = deque(
            GpuBuffer(buffer_size, device) for _ in range(count)
        )

    def _check_budget(self, size: int, existing_bytes: int = 0):
        required = size * self.count * self._copies + existing_bytes
        if required > self.max_owned_bytes:
            raise MemoryError(
                f"Pattern buffer peak would require {required} bytes; "
                f"budget is {self.max_owned_bytes}. Reduce chunk/query size or explicitly raise GPU_SEARCH_BUFFER_BUDGET_MB."
            )

    @property
    def allocated_device_bytes(self) -> int:
        with self._condition:
            return sum(item.allocated_device_bytes for item in self._available) + sum(
                item.allocated_device_bytes for item in self._leases.values()
            )

    def stats(self) -> dict:
        """Pool-owned allocations, including leases; not the whole process/GPU."""
        with self._condition:
            buffers = list(self._available) + list(self._leases.values())
            return {
                "host_staging_bytes": sum(b.capacity for b in buffers),
                "device_bytes": sum(b.allocated_device_bytes for b in buffers),
                "available_buffers": len(self._available),
                "leased_buffers": len(self._leases),
                "closed": self._closed,
                "max_owned_bytes": self.max_owned_bytes,
            }

    def ensure_capacity(self, minimum: int) -> None:
        """Grow all buffers once when an unbounded query needs more overlap."""
        with self._condition:
            if self._closed:
                raise RuntimeError("GPU buffer pool is closed")
            if minimum <= self.buffer_size:
                return
            if self._leased:
                raise RuntimeError("cannot resize the GPU buffer pool while buffers are leased")
            # Replacement is allocated before swapping, so include old + new.
            self._check_budget(minimum, self.buffer_size * self.count * self._copies)
            replacement = deque(
                GpuBuffer(minimum, self.device) for _ in range(self.count)
            )
            self._available = replacement
            self.buffer_size = minimum

    def acquire_buffer(self) -> GpuBuffer:
        with self._condition:
            while not self._available and not self._closed:
                self._condition.wait()
            if self._closed:
                raise RuntimeError("GPU buffer pool is closed")
            buffer = self._available.popleft()
            self._leased += 1
            self._leases[id(buffer)] = buffer
            return buffer

    def release_buffer(self, buffer: GpuBuffer) -> None:
        with self._condition:
            if id(buffer) not in self._leases:
                raise RuntimeError("buffer is not leased from this pool")
            del self._leases[id(buffer)]
            self._leased -= 1
            if not self._closed:
                self._available.append(buffer)
            self._condition.notify()

    @contextmanager
    def acquire(self) -> Iterator[GpuBuffer]:
        buffer = self.acquire_buffer()
        try:
            yield buffer
        finally:
            self.release_buffer(buffer)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._available.clear()
            self._condition.notify_all()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except AttributeError:
            pass
