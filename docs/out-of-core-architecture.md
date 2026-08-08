# Out-of-core exact-search architecture

## Data flow

Repository files (build/update only)
→ packed corpus builder
→ .gpusearch/corpus.bin + files.idx + chunks.idx
→ CandidateSelector (stable chunk IDs)
→ StorageBackend.read(offset, size, GpuBuffer)
→ GpuBufferPool (reusable staging/device allocations)
→ TorchByteSearch.search(buffer, valid_length, prepared_query)
→ stable corpus offset
→ files.idx mapping → file ID, path, file-relative offset and line

The verifier imports neither repository nor storage code. It receives only a device byte tensor, its valid length, and a prepared query. FileStorageBackend is the portable default; MmapStorageBackend and InMemoryStorageBackend implement the same random-access contract.

## Packed format v1

A repository build writes:

- .gpusearch/corpus.bin: file bytes in stable path order, separated by one NUL.
- .gpusearch/files.idx: versioned JSON records containing file ID, repository-relative path, corpus offset, byte length, source signature/hash, and newline offsets.
- .gpusearch/chunks.idx: versioned JSON records containing stable chunk ID, corpus offset, valid length, configured chunk size, and total corpus size.

The NUL separator and result-range validation prevent matches from crossing file boundaries. Original files are not opened during a normal query. They are only read while building or updating the packed corpus.

## Boundary handling

Chunks have non-overlapping primary spans. For a query of length m, a candidate read extends forward by up to m - 1 bytes. Only matches whose start lies in the candidate primary span are accepted. This finds matches crossing any number of chunk boundaries, imposes no fixed query-size limit, and gives every match one owner, so overlap cannot duplicate results. The pool grows once when an unusually long query requires more capacity and reuses that allocation afterward.

## Buffering and pipeline

The pool defaults to two 2 MiB buffers. Each buffer exposes both a writable host view and its device allocation. A storage read returns whether the device allocation is already populated. Current host backends return false, causing one host-to-device copy; a future direct backend can populate the device allocation and return true.

Read, transfer, and verification are separate calls. With two or more buffers, a bounded single-worker pipeline reads chunk N+1 into a leased host buffer while chunk N transfers and verifies. One-buffer configurations retain the synchronous fallback. Device transfer and verification remain on the search thread for safe CUDA/MPS ownership; a future direct backend can add stream/event coordination without changing storage, selection, mapping, or the verifier API. Query metrics expose whether the pipeline ran, how many chunks were prefetched, and how many bytes arrived device-ready without a search-engine staging copy. Device-ready bytes are not, by themselves, proof that a transport avoided an internal compatibility bounce buffer.

## Candidate selection

CandidateSelector.select(query, catalog) returns stable chunk IDs. The default selector returns every chunk. An optional `TrigramCandidateSelector` builds an in-memory posting map from the packed corpus and selects chunks containing the query's first ASCII-folded trigram. It indexes a two-byte overlap so a trigram starting at a chunk boundary remains owned by the correct primary chunk. Queries shorter than three bytes conservatively select every chunk, and exact-case searches may receive harmless false positives but never false negatives. The current trigram map is rebuilt from packed bytes at index startup; persisting it is a future optimization. Bloom-filter, prefix, semantic, or other indexes can replace it without changing verification.

## Metrics

GpuFileIndex.stats()["last_query"] reports corpus/chunk/candidate counts, candidate percentage, physical storage bytes and percentage, bytes made available to the GPU, host-to-device bytes/time, storage time, verification time, total latency, and reusable-buffer VRAM. Build stats include corpus size, source bytes, file/chunk counts, chunk size, and build time. The benchmark runner prints these values per query.

Physical bytes may exceed 100% for an all-chunk scan because dynamic overlap and small result-line reads are included. This is intentional and makes the metric useful when candidate pruning is introduced.

## Adding KvikIO next

The tested transport contract is ready, but no KvikIO or GDS implementation or dependency is included. See [the KvikIO backend plan](kvikio-backend-plan.md) for the implementation and validation gates.

1. Add an optional KvikIOStorageBackend implementation; do not make RAPIDS a core dependency.
2. Accept the packed corpus path and keep the same byte address space.
3. Implement read(offset, size, destination) using the destination device allocation and return ReadResult(size, device_ready=True).
4. Enforce and report deployed alignment requirements, using a reusable aligned bounce region only when unavoidable.
5. Add CUDA stream/event integration to the pool, then prefetch chunk N+1 while the verifier handles chunk N.
6. Run equivalence, unaligned-tail, short-read, cancellation, and throughput tests against the existing backends.

## Adding native cuFile/GDS later

A native backend additionally needs cuFile driver/runtime detection, file-handle registration, device-buffer registration (or documented implicit registration), alignment-aware fast paths plus correct unaligned head/tail handling, error and fallback policy, CUDA stream/event ownership, and cleanup ordering. Packaging must keep it optional and platform-gated. The backend should still return stable byte counts and `device_ready=True`; query parsing, candidate indexes, result mapping, and `TorchByteSearch` remain unchanged. See the [native cuFile/GDS backend plan](native-cufile-gds-plan.md) for lifecycle, deployment, observability, and validation gates.

SCADA-style or other direct transports follow the same contract: implement positioned reads into the reusable destination and expose no transport details to the verifier.
