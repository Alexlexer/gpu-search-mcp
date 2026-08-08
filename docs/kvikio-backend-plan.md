# Optional KvikIO backend plan

This document defines the next storage-transport step. It does **not** add KvikIO,
cuFile, or GPUDirect Storage code or dependencies.

## Existing contract

`StorageBackend.read(offset, size, destination)` owns transport details. A normal
host backend writes `destination.host_view` and returns
`ReadResult(bytes_read, device_ready=False)`. A direct transport may instead write
the reusable `destination.device_buffer` and return `device_ready=True` only after
those bytes are safe for the verifier to consume. The search loop then skips the
host-to-device copy. Result addresses remain corpus offsets, never device pointers.

A contract test uses a test-only device-ready backend to exercise this path without
pretending that GDS is available. Query metrics separately report
`direct_storage_bytes`, `host_to_gpu_bytes`, and total bytes made available to the
GPU.

## Proposed optional backend

1. Put `KvikIOStorageBackend` in an optional module and import `kvikio` lazily.
   The base install and current `file`, `mmap`, and `memory` backends must remain
   unchanged.
2. Open `.gpusearch/corpus.bin` once with `kvikio.CuFile` and preserve the packed
   corpus's existing offset space. Do not reopen repository source files.
3. For a pool destination, issue a positioned `CuFile.pread` into a buffer object
   that exposes the Torch allocation through a supported CUDA array/buffer
   protocol. Confirm the exact adapter against the selected Torch, CUDA, and
   KvikIO versions rather than relying on an undocumented raw pointer.
4. Keep the future returned by `pread` owned by the read operation and call
   `get()` before returning `ReadResult(..., device_ready=True)` in the first
   synchronous implementation. This makes the current contract truthful. A later
   contract revision may carry a completion token/event for overlap.
5. Treat short reads as byte counts. Existing pool validation will convert an
   unexpected short chunk read into a clear `EOFError`.
6. Continue using host destinations for small result-line reads. The backend must
   return `device_ready=False` for those reads.
7. Close the `CuFile` handle after all in-flight reads finish. If explicit device
   memory registration is enabled, register each long-lived pool allocation once
   and deregister it before freeing the allocation.

## Runtime policy

KvikIO has its own compatibility setting: `AUTO` attempts cuFile and falls back to
POSIX, `OFF` requires cuFile, and `ON` avoids loading `libcufile`. The backend should
expose an explicit application policy rather than silently labeling compatibility
I/O as GDS:

- `auto`: permit KvikIO fallback, but report the active mode and do not count
  compatibility reads as verified direct-storage bytes until KvikIO exposes a
  reliable per-read signal.
- `require-gds`: fail during backend creation if direct cuFile I/O cannot be
  established.
- `disabled`: use an existing portable backend without importing KvikIO.

The default should remain `disabled` until supported Linux/NVIDIA hardware is in
CI or a dedicated benchmark host.

## Alignment and tails

The packed format should remain unpadded until measurements justify a format
change. KvikIO documents support for unaligned operations, with lower performance
than aligned transfers, and its opportunistic direct-I/O settings can handle
unaligned POSIX portions. Benchmark 4 KiB-aligned chunk sizes and corpus offsets,
but keep final partial chunks and arbitrary query overlap correct. Alignment and
bounce-buffer decisions belong entirely inside the backend.

## Required validation

- Backend conformance: bounds, zero-length and short reads, host destinations,
  device destinations, cleanup, and repeated pool reuse.
- Search equivalence across chunk starts, chunk ends, long overlap, final partial
  chunks, binary data, and adjacent packed files.
- Mode reporting for KvikIO compatibility, cuFile compatibility, and confirmed GDS.
- Failure tests for missing optional packages, missing `libcufile`, unsupported
  filesystems, and mid-query I/O errors.
- CUDA synchronization tests proving that `device_ready=True` is never returned
  before the verifier can safely consume the allocation.
- Cold-cache and warm-cache benchmarks for throughput, latency, CPU use, physical
  bytes read, direct-storage bytes, H2D bytes, and buffer-pool VRAM.

## Upstream references

- [KvikIO Python documentation](https://docs.rapids.ai/api/kvikio/stable/)
- [KvikIO quickstart and positioned reads](https://docs.rapids.ai/api/kvikio/stable/quickstart/)
- [KvikIO runtime and compatibility settings](https://docs.rapids.ai/api/kvikio/stable/runtime_settings/)
- [libkvikio file-handle API](https://docs.rapids.ai/api/libkvikio/stable/classkvikio_1_1filehandle)

Version-pin the optional extra and revalidate this plan against the selected stable
KvikIO release when implementation begins; the referenced API is currently the
26.06 stable documentation.
