# Runtime safety and memory visibility

`GET /runtime` reports process RAM, private committed memory on Windows, process
CPU seconds, Python thread count, initialized PyTorch CUDA allocations/reservations,
the bounded watcher queue counters, and pattern-buffer host/device allocations.
It does not initialize indexes, load a model, start workers or scan the repository.
Missing metrics are `null` with warnings where appropriate, never invented zeros.

Windows uses [GetProcessMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo)
and [PROCESS_MEMORY_COUNTERS_EX](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters_ex).
Private committed bytes are not resident RAM. Linux reports RSS/HWM from procfs;
live process RAM on other platforms is currently unavailable. PyTorch metrics
exclude allocations made outside its allocator and do not describe total GPU use.
Python thread counts exclude native-only threads. These snapshots are observability,
not a process-wide memory cap or proof that no leak remains.

Pattern buffers are counted while checked out, including after pool close until
their leases return. Closing releases pool ownership; callers retaining references
or allocator caches can keep actual memory resident longer. Do not add the pool
figures to process/PyTorch totals: those totals already overlap.

Watcher counters retain no source text or event history: pending/capacity, running,
submitted/coalesced/completed/failed, high-water mark and backpressure waits.
Completed includes failed callbacks. A rising upstream filesystem-event backlog is
not measured by these downstream queue counters yet.

Safe rollout: inspect scope first, then test a small disposable repository under
memory/process limits before restoring an unrestricted production root. Sample
`/runtime` during changes and after settling; check that threads, queue depth and
rebuild activity settle. Full-workload memory-growth validation remains necessary.

## Pattern buffer allocation budget

`GPU_SEARCH_BUFFER_BUDGET_MB` is a positive integer, default **64 MiB**. It limits
logical pool-owned host staging plus device buffers, not the entire process or
semantic model. CPU staging/device aliases count once. Accelerator buffers count
both allocations. Resizing checks the temporary **old + replacement** peak before
allocating; rejection leaves the old pool usable. A clear `MemoryError` replaces
an attempted unbounded buffer allocation. Raise the limit only deliberately.

Allocator rounding/reservations, externally retained references, PyTorch caches,
file metadata, semantic text, embeddings and upstream watcher queues are **not**
covered by this budget. Process-wide RAM/VRAM admission control remains follow-up
work; 64 MiB here must never be presented as a total service memory guarantee.
