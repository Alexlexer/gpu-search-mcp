# Current project state

Last reconciled: 2026-08-10

This document is the short, implementation-grounded snapshot of gpu-search-mcp. It exists so contributors and coding agents do not infer the current architecture from older benchmark or README wording.

## Product direction

gpu-search-mcp is evolving from GPU-accelerated repository search into a **local-first code-intelligence/context engine for AI coding agents**.

The product goal is not merely faster grep. A high-level request should be able to return a compact, explainable bundle of the relevant implementation, symbols, callers, dependencies, configuration, tests, Git context, risks, unknowns, likely change set, and recommended inspection order.

GPU acceleration is an implementation advantage. Correct CPU behavior remains mandatory, and source code should be able to remain entirely on the local machine/private worker.

## Current authoritative runtime

The supported implementation is Python-only.

The abandoned Rust rewrite is historical work and is not part of the active architecture or roadmap. New work must not reintroduce a Rust core, Rust sidecar, Cargo workspace, or migration track unless the project direction is explicitly changed first.

## What is implemented

### Exact search

Exact pattern search is out-of-core.

Data flow:

```text
repository files (build/update only)
        |
        v
.gpusearch/corpus.bin
.gpusearch/files.idx
.gpusearch/chunks.idx
        |
        v
CandidateSelector
        |
        v
StorageBackend
        |
        v
GpuBufferPool
        |
        v
TorchByteSearch
        |
        v
stable corpus/file offsets
        |
        v
existing result contracts
```

Key properties:

- Source files are streamed into a versioned packed corpus.
- Normal queries do not reopen original source files.
- The default chunk size is 2 MiB.
- The default reusable buffer count is two.
- Exact-search device working memory is bounded by the reusable buffers rather than repository size.
- `TorchByteSearch` is storage-agnostic.
- File/chunk boundary handling supports matches spanning chunk boundaries while rejecting matches spanning file boundaries.
- Case-insensitive matching no longer requires a full lowercase corpus copy.
- `FileStorageBackend`, `MmapStorageBackend`, and an explicit in-memory backend implement the same random-access transport contract.
- The storage contract leaves a future seam for optional KvikIO/cuFile/direct-storage implementations.

### Current exact-search limitation

A conservative first-trigram selector and versioned persistent posting index are implemented behind `CandidateSelector`. `AllChunksCandidateSelector` remains the production-safe baseline, and exact verification remains authoritative. Missing, stale, invalid, or corrupt candidate indexes rebuild safely.

The remaining retrieval question is quantitative scale: candidate selectivity and physical reads have only been measured on small corpora. The planned 1/10/30/100 GiB benchmark must establish where first-trigram filtering is sufficient before rarer-trigram intersection or storage/GPU pipeline work is attempted.

### Semantic retrieval

Implemented with sentence-transformers and a persistent embedding cache. Semantic indexing/model availability is optional and degrades explicitly when unavailable.

### C# symbol intelligence

Milestone 2 is complete.

The Python symbol graph exposes stable symbols/edges and operations for:

- symbol lookup
- references
- implementations
- callers
- callees
- tests
- impact explanation

C# extraction covers common declarations and relationships including ASP.NET endpoints, DI, inheritance/implementation, construction/references, overrides, and tests. Confidence/provenance are part of the model. It remains heuristic rather than Roslyn/compiler-accurate.

### Change planning

Milestone 3 is complete.

`ChangePlanner` combines exact, semantic, symbol, dependency, and Git evidence into deterministic token-budgeted change plans containing:

- primary implementation
- parent context
- direct callers
- direct dependencies
- implementations/overrides
- configuration/documentation
- tests/coverage
- Git context
- reasons/confidence
- omissions
- risks
- unknowns
- likely change set
- inspection order

This is currently the part of the repository most closely aligned with the long-term agent-context product thesis.

### Quality/reliability

Implemented foundations include:

- versioned retrieval-quality manifests
- Recall@1/5/10
- Precision@5
- mean reciprocal rank
- exact-symbol recall
- related-test recall
- returned-token measurements
- CPU quality regression gates
- package/smoke validation
- content-addressed cache identities
- repository locks
- stale-lock recovery
- temporary staging + fsync
- atomic promotion
- rollback/recovery coverage

The current benchmark fixtures are useful regression tests but are not yet broad enough to substantiate product-level claims about agent effectiveness.

## Current priority order

### NOW — agent evaluation harness

Establish a reproducible baseline comparing the same coding agent with and without the current GPU Search tools. Capture task outcome, validation, patch, repository exploration, token usage when available, timing, and sanitized trajectories. The harness must remain opt-in and normal CI must use deterministic fake adapters only.

The initial real task corpus should focus on realistic C#/.NET work, but building a statistically meaningful 20-task corpus is separate from the instrumentation foundation.

### NEXT — promote agent context surface

Evaluate whether `plan_change` should be extended or wrapped by a stable high-level `prepare_context`-style operation.

The high-level API should expose structured implementation, symbols, relationships, tests, configuration, Git state, risks, unknowns, omissions, confidence/provenance, and inspection order without forcing agents to manually chain many low-level retrieval calls.

### LATER — improve C# intelligence from benchmark failures

Use failures from the agent harness to prioritize structural improvements rather than adding unmeasured heuristics.

Potential areas include more accurate ASP.NET routing, DI, options/configuration binding, EF/DbContext relationships, MediatR handlers, extension methods, partial classes, and call relationships.

### LATER — asynchronous storage/GPU pipeline

The buffer pool currently provides reusable allocations, but the search loop is synchronous. After candidate pruning, profile whether CUDA streams/events and double-buffered prefetch materially improve remaining latency.

### LATER — KvikIO/cuFile/GDS

Direct storage is an optional backend optimization, not a near-term product requirement.

The current `StorageBackend`/destination contract is the integration seam. Do not redesign search, candidate selection, result addressing, MCP, or HTTP around GDS.

### LATER — persistent private worker and SaaS control plane

Only after agent value is measured should the local runtime evolve into a multi-repository worker and optional SaaS control plane.

Source/indexing/retrieval should remain local/private by default; cloud services can later coordinate identity, organizations, policy, worker pairing, audit, usage, GitHub integration, and billing.

## Documentation notes

Some historical performance claims and release text may predate the out-of-core refactor. When documentation conflicts, prefer:

1. current code and tests
2. `docs/out-of-core-architecture.md`
3. `docs/benchmarks/out-of-core-baseline-2026-08-08.md`
4. this project-state document
5. older benchmark/README prose

Documentation should be reconciled before making new product/performance claims.
