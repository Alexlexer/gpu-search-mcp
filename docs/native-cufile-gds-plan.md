# Native cuFile / GPUDirect Storage backend plan

This is an implementation plan and acceptance checklist, not a GDS implementation.
GPU Search must continue to work without CUDA, `libcufile`, `nvidia-fs`, or a
GDS-capable filesystem.

## Architectural boundary

A future `CuFileStorageBackend` implements only the positioned byte transport:

```text
Candidate chunk (corpus offset + valid length)
  -> StorageBackend.read(offset, size, reusable destination)
  -> cuFile transport fills destination.device_buffer
  -> ReadResult(bytes_read, device_ready=True)
  -> unchanged TorchByteSearch(buffer, valid_length, query)
  -> unchanged stable corpus-offset result mapping
```

It must not parse queries, choose chunks, interpret paths, map results, or call the
verifier. The backend opens only `.gpusearch/corpus.bin`; normal search must never
open original repository files.

`device_ready=True` means the verifier may safely consume the allocation without a
search-engine host-to-device copy. It does **not** prove GDS: cuFile compatibility
mode can internally use a host bounce buffer and still leave the final bytes on the
GPU. Keep `device_ready_bytes` as the transport-neutral metric. Add a separate
`verified_gds_bytes` metric only when the backend runs with compatibility fallback
disabled or has another reliable runtime proof of the direct path.

## Recommended implementation boundary

Use a small optional native extension (C++/CUDA or a deliberately narrow binding)
to own cuFile resources. Do not spread raw handles or device pointers through Python.
The extension should expose a minimal object such as:

```text
open(corpus_path, device_id, mode_policy)
register_buffer(base_pointer, allocation_size) -> buffer_token
read(file_offset, size, buffer_token, buffer_offset=0) -> bytes_read
close()
```

Python adapts that object to `StorageBackend`. The binding must translate all cuFile,
CUDA, `errno`, and short-read outcomes into typed exceptions or byte counts while
retaining the original error code and operation context.

Torch interoperability is the main early proof point. Confirm that the Torch CUDA
allocation is compatible with cuFile registration, stays alive and at a stable base
address for the entire pool lifetime, belongs to the current CUDA context/device,
and is not resized while registered. If supported allocation types cannot be
registered, allocate the pool through a compatible CUDA allocator and expose it to
Torch without copying; do not add a hidden per-chunk bridge allocation.

## Synchronous lifecycle (first implementation)

1. Probe platform and optional library availability without importing or loading
   cuFile in the base path.
2. Open the packed corpus with the flags required by the selected deployment policy.
   Register the file descriptor with `cuFileHandleRegister` and retain both the OS
   descriptor and `CUfileHandle_t`.
3. Allocate the fixed GPU buffer pool once. Optionally call `cuFileBufRegister` once
   per allocation after measuring explicit versus implicit registration. Registered
   reads must keep the registered base pointer and express subranges through the
   buffer offset.
4. Implement a blocking `read` with `cuFileRead`. Return the exact successful byte
   count and set `device_ready=True` only after the call completes.
5. On shutdown, stop submissions, wait for every in-flight operation, deregister
   buffers, release GPU allocations, deregister the cuFile handle, close the file
   descriptor, and finally close the cuFile driver/runtime. Cleanup must be
   idempotent and safe after partial initialization.

The first native version should remain synchronous. It proves correctness and
resource ownership before introducing stream-ordered I/O.

## Async and double-buffering phase

The current two-buffer pipeline separates reads from verification but its Python
read worker is not enough to prove CUDA ordering. A later native phase may use the
cuFile stream APIs or batch APIs:

- Associate completion with the exact pool lease; never return a buffer to the pool
  while I/O or verification still references it.
- Register/deregister CUDA streams according to the cuFile API and make ownership
  explicit per device/context.
- Use CUDA events or an equivalent completion token so verification waits for read
  completion and the next read waits for prior buffer use.
- Keep scalar arguments and completion byte-count storage alive for the full async
  operation; the stream APIs defer access to some pointer arguments.
- Handle cancellation and teardown by draining or cancelling submissions before
  destroying batch, stream, file, or buffer resources.
- Measure the stream APIs against blocking reads plus the existing double buffer.
  Keep the simpler path if it performs as well for 2 MiB chunks.

No async API should weaken the core rule: the verifier receives only a byte buffer,
valid length, and query.

## Alignment and packed-corpus policy

Current cuFile documentation says synchronous reads support unaligned offsets and
sizes, though aligned I/O can perform better. Therefore:

- Correctness must not depend on 4 KiB alignment.
- Preserve final partial chunks and arbitrary query overlap exactly.
- Start with the existing default 2 MiB chunk size, which is page-multiple sized.
- Benchmark aligned corpus offsets, registered buffer bases, and transfer sizes.
- Let the backend handle unaligned corpus starts/tails; do not pad `corpus.bin` or
  change stable file offsets until benchmark evidence justifies a packed-format
  version change.
- If a bounce region is required, allocate it once per pool buffer and report its
  bytes and time. Never allocate it per chunk.

Avoid mixing overlapping normal buffered I/O or `mmap` access with direct I/O on the
same file during an active query. Host result-line materialization should either use
cuFile host reads under the same handle policy or a clearly separated safe path
validated on the target filesystem.

## Deployment and capability gates

A supported deployment is more than finding `libcufile.so`. Startup diagnostics
must capture:

- Linux, NVIDIA driver, CUDA toolkit/runtime, cuFile, and (where applicable)
  `nvidia-fs` versions.
- GPU model, compute capability, CUDA device/context, and peer topology.
- Packed-corpus mount, filesystem, block/NVMe or network transport, and whether that
  path is GDS capable and not denied by configuration.
- `cufile.json` location and effective compatibility/fallback policy.
- Relevant IOMMU/ACS, container device/mount, and vendor filesystem prerequisites.
- `gdscheck` results on deployment/benchmark hosts; do not execute privileged probes
  inside each query.

Recent NVIDIA documentation describes a P2PDMA path for some CUDA 12.8+ local-NVMe,
Linux-kernel, driver, and GPU combinations that may not require `nvidia-fs`. Treat
that as a detected platform capability, not a universal requirement or assumption.
Always validate the exact current NVIDIA support matrix for the target host.

## Mode and fallback policy

Expose an explicit, observable policy:

- `disabled` (default): use `FileStorageBackend` or `MmapStorageBackend`; do not load
  cuFile.
- `prefer-gds`: try the native backend and fall back before query execution if the
  platform probe or handle registration fails. Record the selected backend and
  reason. Do not change backend halfway through a chunk silently.
- `require-gds`: disable compatibility fallback and fail fast unless the corpus
  handle and device buffers can use the intended GDS path.

A successful `cuFileRead` is not sufficient evidence of direct DMA when compatibility
mode is allowed. Benchmarks and CLI output must distinguish requested policy,
selected backend, device-ready delivery, and verified GDS delivery.

## Error and cleanup requirements

- Reject negative offsets/sizes and destination overflow before entering native code.
- Preserve short reads as counts; the existing chunk loader rejects unexpected EOF.
- Include operation, corpus offset, requested size, device, filesystem, cuFile error,
  and `errno` where applicable in diagnostics without leaking repository contents.
- Define retry policy only for documented transient outcomes. Never retry arbitrary
  reads after a partial completion without advancing offsets correctly.
- Make file, buffer, stream/batch, and driver ownership exception-safe.
- Prevent close while buffers are leased or I/O is in flight; repeated close is safe.
- A failed optional backend must not corrupt the packed corpus or poison the portable
  backend used by the next query.

## Packaging and build

- Keep the native extension in an optional extra/wheel variant; base wheels remain
  portable and must not link to cuFile.
- Gate builds by supported Linux/CUDA targets and publish the compatibility matrix.
- Prefer runtime loading or a separately packaged extension so importing GPU Search
  on unsupported hosts still works.
- Pin and test the CUDA/cuFile ABI combinations used to build wheels.
- Add license notices and redistribution checks for headers and runtime libraries.
- Do not make RAPIDS/KvikIO a dependency of the native backend.

## Validation matrix

### Functional equivalence

Run the existing legacy/out-of-core matrix plus native-backend cases for empty and
binary files, UTF-8, adjacent packed files, zero/many matches, chunk starts/ends,
cross-boundary matches, query lengths larger than a chunk, duplicate suppression,
final partial chunks, candidate pruning, and stable result-to-file mapping.

### Backend conformance

Test bounds, zero-length reads, short reads, destination overflow, registered and
implicit buffers, buffer offsets, repeated reuse, multi-device context rejection,
concurrent sessions, failure injection, partial initialization, and cleanup ordering.
Use a test double only for contract tests; label it device-ready, never GDS.

### Hardware integration

On each supported host/filesystem combination:

1. Run NVIDIA `gdscheck` and record versions/topology/configuration.
2. Run NVIDIA data-integrity verification tooling before application benchmarks.
3. Search a deterministic packed corpus with portable and native backends and compare
   byte-for-byte results.
4. Exercise aligned and unaligned offsets/sizes, final tails, cold/warm cache, one and
   multiple buffers, and synchronous/async variants.
5. Verify direct-path counters/telemetry where available and prove that
   `require-gds` does not fall back.
6. Repeat under sustained load and teardown/rebuild loops to catch registration and
   lifetime faults.

### Performance acceptance

Report total corpus size, chunk/candidate counts, physical bytes read, corpus-read
ratio, device-ready bytes, verified GDS bytes, H2D bytes, read/H2D/kernel/total time,
CPU utilization, throughput, tail latency, pool VRAM, registration time, and any
internal bounce bytes. Compare against `file` and `mmap` on the same mount and cache
state. Do not claim a GDS win based only on warm page-cache tests.

Promote the backend from experimental only when it preserves exact results, has no
per-chunk allocation/registration, makes fallback visible, and demonstrates a
repeatable benefit on at least one documented supported configuration without a
material regression on realistic selective-query workloads.

## KvikIO versus native cuFile

| Concern | Optional KvikIO backend | Native cuFile backend |
|---|---|---|
| Integration effort | Lower; Python/C++ wrapper already provided | Higher; own binding and resource lifecycle |
| Dependencies | Optional RAPIDS package | Optional CUDA/cuFile native extension |
| Low-level control | KvikIO policy and exposed APIs | Direct handle, buffer, stream, and batch control |
| Compatibility fallback | KvikIO and cuFile layers must both be reported | Application can require cuFile with fallback disabled |
| Best first use | Validate device-buffer interoperability and throughput | Deploy when explicit lifecycle/telemetry/control justifies complexity |

Prototype KvikIO first on the target Linux/NVIDIA host unless a native-only feature
is required. Keep both implementations behind the same `StorageBackend` contract so
the search engine, candidate selectors, verifier, and result mapper never diverge.

## Official references

- [NVIDIA GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/index.html)
- [Getting Started with GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/getting-started/)
- [cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
- [GPUDirect Storage Overview Guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)
- [O_DIRECT Requirements Guide](https://docs.nvidia.com/gpudirect-storage/o-direct-guide/index.html)
- [Best Practices Guide](https://docs.nvidia.com/gpudirect-storage/best-practices-guide/index.html)
- [Installation and Troubleshooting Guide](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/)

Revalidate all platform requirements, supported filesystems, APIs, and known issues
against the NVIDIA documentation shipped with the exact CUDA release selected for
implementation. This plan was checked against the current NVIDIA documentation on
2026-08-08.
