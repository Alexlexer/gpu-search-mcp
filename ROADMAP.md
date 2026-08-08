# gpu-search-mcp Python-Only Roadmap

> Status: Active
> Last reconciled: 2026-08-08
> Direction: Python-only. No Rust core, bindings, sidecars, Cargo workspace, or Rust migration work.

For the implementation-grounded snapshot, see [`docs/project-state.md`](docs/project-state.md).

## Product outcome

Build a production-ready, local-first code-intelligence/context engine for AI coding agents. One request should return a compact, explainable bundle of the relevant implementation, symbols, callers, dependencies, configuration, tests, Git context, risks, unknowns, likely change set, and recommended inspection order.

The product is more than GPU grep or vector search. GPU acceleration is optional; correct and useful CPU-only behavior is mandatory. Source code should be able to remain entirely on the local machine or private worker.

## Principles

- **Local-first:** no account, API key, cloud database, or telemetry required for normal operation.
- **Agent-first:** compact, structured, explainable, change-oriented responses.
- **Progressive capability:** exact search works without embeddings; optional capabilities degrade explicitly.
- **Stable contracts:** MCP and HTTP remain backward-compatible while versioned surfaces are added.
- **Python-only:** do not reintroduce the abandoned Rust rewrite.
- **Secure by default:** canonical paths, local binding, redaction, resource limits, and safe diagnostics.
- **Evidence-based:** performance and retrieval claims require reproducible benchmarks.
- **Small slices:** every change includes affected files, tests, validation, and known limitations.

## Baseline to preserve

- PyTorch exact byte-pattern verification.
- CUDA, Apple Silicon MPS, and CPU execution.
- Out-of-core packed exact-search corpus with bounded reusable buffers.
- Replaceable `StorageBackend` transport.
- `CandidateSelector` abstraction.
- Sentence-transformer semantic retrieval and hybrid search.
- Persistent semantic/dependency/cache metadata and packed exact-search artifacts.
- Filesystem watching and update handling.
- MCP stdio and local HTTP transports.
- Dependency impact analysis.
- AST block expansion and file skeletons.
- Language-neutral symbol graph with C#/.NET intelligence.
- Deterministic `plan_change` context bundles.
- Secret redaction and indexed-root validation.
- LegacyLens signal scanning.
- Console entry point, setup/doctor commands, smoke tests, packaging, and structured HTTP results.

Any replacement must match current behavior before an old field, route, command, or tool is deprecated.

## Current architecture status

### Exact search — out-of-core implemented

Current data flow:

```text
repository files (build/update only)
        |
        v
.gpusearch/corpus.bin + files.idx + chunks.idx
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

Implemented:

- versioned packed corpus
- stable file/chunk addressing
- NUL separators + result-range validation to prevent cross-file matches
- configurable chunk size (2 MiB default)
- reusable bounded host/device buffers (two by default)
- file, mmap, and explicit memory backends
- storage-agnostic Torch verifier
- query-length-aware overlap without a fixed small query-length limit
- equivalence coverage across tiny chunk sizes, UTF-8, NUL bytes, overlap, long queries, missing results, and `max_files`
- per-query candidate/read/transfer/kernel/latency/VRAM metrics

The previous full raw + lowercase GPU corpus is no longer the active design.

### Main exact-search gap — candidate pruning

The current `AllChunksCandidateSelector` returns every chunk.

Therefore out-of-core search removes the VRAM-size ceiling but still tends toward O(corpus-size) reads and verification per exact query.

The initial 64 MiB CUDA baseline showed:

- 4 MiB reusable-buffer VRAM
- ~32x less measured corpus-related VRAM than the previous resident implementation
- ~2.6x faster clean packed-corpus build on that workload
- 28–46% slower dense all-chunk queries
- ~100% candidate percentage / ~1.0 physical-read ratio

The next scaling feature is selective candidate indexing, not GDS.

## Delivery map

| Milestone | Outcome | Status | Exit gate |
|---|---|---:|---|
| 1. Usable local product | Unified search, setup, diagnostics, packaging, onboarding | Mostly complete | Fresh CPU install configures a client and completes a diagnostic search |
| 2. C# intelligence | Language-neutral symbol graph and useful C# relationships | Completed | C# fixtures pass symbol, caller, DI, endpoint, implementation, and test queries |
| 3. Change planning | Token-budgeted plans with risks and inspection order | Completed | Change requests return implementation, impact, config, tests, omissions, and risks |
| 4. Quality/reliability | Benchmarks, regression gates, reliable caches, out-of-core exact search | In progress | CI + runtime evidence detect quality/resource regressions and exact search scales beyond VRAM |
| 5. Candidate pruning + agent evaluation | Selective exact retrieval and measured coding-agent value | NOW / NEXT | Selective queries read a small corpus fraction and agent benchmark demonstrates value |
| 6. Context productization | Stable high-level agent-context API and stronger .NET intelligence | Planned | One request reliably provides the context required for realistic .NET changes |
| 7. Worker/distribution | Persistent multi-repo private worker | Planned | One worker safely serves multiple isolated repositories and agent sessions |
| 8. Security/public API | Versioned API, authentication/limits for non-local transport | Planned | Security and transport end-to-end matrices pass |
| 9. Optional SaaS | Control plane around private workers | Later | Team coordination works without requiring source upload |

# Milestone 1 — Usable local product

Substantially implemented:

- unified `search_code` request/response contract
- explicit intent and mode normalization
- exact/semantic/hybrid/symbol retrieval surfaces
- setup workflow for Codex/Claude
- read-only doctor command
- optional package extras
- pipx / uv tool / uvx support
- wheel + sdist build
- isolated outside-checkout package smoke
- local HTTP + MCP transports
- root isolation and safe diagnostics

Remaining cleanup should be driven by fresh-install validation rather than reimplementing completed packaging work.

# Milestone 2 — C# symbol intelligence

Status: completed on 2026-07-22.

Implemented:

- stable Python `Symbol` / `SymbolEdge` graph with deterministic identifiers
- C# declarations and signatures
- imports/references/calls
- inheritance and implementation relationships
- instantiation and overrides
- ASP.NET endpoints/controllers
- dependency injection relationships
- test relationships
- `find_symbol`
- `find_references`
- `find_implementations`
- `find_callers`
- `find_callees`
- `find_tests`
- `explain_impact`
- confidence/provenance metadata

The implementation remains heuristic rather than Roslyn/compiler-accurate.

# Milestone 3 — Agent change planning

Status: completed on 2026-07-22.

`plan_change(request, top_k, max_context_tokens)` creates deterministic token-budgeted bundles containing:

1. Primary implementation.
2. Parent class/module context.
3. Direct callers.
4. Direct dependencies.
5. Implementations/overrides.
6. Related configuration/documentation.
7. Tests and missing coverage.
8. Relevant Git changes.
9. Match reasons/confidence.
10. Risks, unknowns, omissions, likely change set, and inspection order.

Git state may boost ranking but must not outrank an exact symbol match.

# Milestone 4 — Retrieval quality, reliability, and out-of-core search

Status: in progress.

Completed retrieval-quality foundation:

- versioned JSON/YAML benchmark manifests
- C#, TypeScript, Python, and mixed fixtures
- Recall@1/5/10
- Precision@5
- mean reciprocal rank
- exact-symbol recall
- related-test recall
- returned-token measurements
- ripgrep/exact/symbol/semantic/hybrid comparison modes
- portable CPU baselines
- CI zero-quality-drop gates
- bounded token-growth gate
- hard compact-output ceiling

Completed cache/reliability foundation:

- SHA-256 source-content identities
- schema/app/parser/model/chunking/configuration identities
- repository cache locks
- stale-lock recovery
- temporary staging
- fsync
- atomic promotion
- rollback backups
- interrupted-transaction recovery
- failure-injection coverage

Completed out-of-core foundation:

- `.gpusearch/corpus.bin`
- `files.idx`
- `chunks.idx`
- `PackedCorpusCatalog`
- `StorageBackend`
- `GpuBufferPool`
- `TorchByteSearch`
- `CandidateSelector`
- bounded exact-search VRAM
- result/boundary equivalence tests
- out-of-core instrumentation and CUDA baseline

Remaining reliability work:

- branch/worktree/rename/watcher-storm reconciliation coverage
- runner-specific latency gates where reproducible
- keep update/rebuild semantics safe under concurrent search and watcher activity

# Milestone 5 — Candidate pruning + agent evaluation

## 5A. Candidate chunk index — NOW

Goal: make out-of-core exact search selective.

Evaluate an initial index such as trigram/ngram postings or compressed chunk bitmaps behind the existing `CandidateSelector` contract.

Requirements:

- zero false negatives for supported exact-query semantics
- deterministic chunk IDs
- handle short queries explicitly
- preserve case-sensitive/case-insensitive semantics
- preserve chunk/file-boundary correctness
- measure index size, build time, update cost, and query pruning
- preserve `AllChunksCandidateSelector` as correctness/baseline mode

Exit gate:

- parity with all-chunks results
- materially lower candidate percentage on selective queries
- materially lower physical read ratio
- no unmeasured memory explosion in the candidate index

## 5B. Agent evaluation harness — NEXT

Primary product hypothesis:

> A coding agent using gpu-search-mcp should solve software-engineering tasks with less irrelevant context and fewer unnecessary file reads/tool calls while maintaining or improving correctness.

Start with realistic C#/.NET tasks, eventually 30–50 tasks across categories such as:

- bug fixes
- endpoint additions
- interface changes
- DI changes
- configuration changes
- validation changes
- EF/data-layer changes
- test updates
- dependency changes
- multi-file refactors

Compare:

```text
coding agent alone
vs
coding agent + gpu-search-mcp
```

Measure where practical:

- task success
- tests passing
- patch correctness
- files inspected
- irrelevant files inspected
- retrieval/tool calls
- input/context tokens
- output tokens
- time to first relevant file
- time to final patch

Deterministic retrieval/fixture metrics can run in CI; full agent runs should initially be offline/nightly because model behavior, cost, and latency are not deterministic enough for a strict per-commit gate.

# Milestone 6 — Context productization

Evaluate whether `plan_change` should be extended or wrapped by a stable `prepare_context`-style operation.

High-level output should include:

- primary implementation
- relevant symbols
- callers/callees where useful
- dependencies
- implementations/overrides
- tests
- configuration
- Git context
- risks
- unknowns
- likely change set
- inspection order
- token-budget omissions
- confidence/provenance

Keep lower-level search tools available.

Ranking improvements must be driven by failures observed in the agent evaluation harness rather than arbitrary scoring constants.

For C#, prioritize structural improvements that measurably improve agent tasks, potentially including:

- ASP.NET routing
- DI registrations/lifetimes
- options/configuration binding
- EF `DbContext` and entities
- MediatR handlers
- extension methods
- partial classes
- higher-confidence call relationships

Do not expand to many languages before the .NET path is demonstrably strong.

# Milestone 7 — Persistent private worker

Evolve the local process into a persistent worker only after the engine proves useful.

Target capabilities:

- stable worker identity
- multiple repositories
- strict repo isolation
- persistent cache lifecycle
- bounded memory
- device/resource reporting
- indexing status
- background updates
- version reporting
- agent sessions
- authentication for non-stdio access
- structured logs
- resource limits

Source remains local/private by default.

# Milestone 8 — Security and versioned public API

Potential versioned routes:

- `/v1/search/code`
- `/v1/search/symbol`
- `/v1/change/plan`
- `/v1/context/prepare`
- `/v1/index/root`
- `/v1/index/status`
- `/v1/diagnostics`

Security work:

- canonicalize every path under indexed roots
- reject traversal/outside-root/symlink/case/UNC/encoded bypasses
- bound repository/file/query/result/context/concurrency/semantic-batch sizes
- redact credentials/tokens/passwords/connection strings/private keys/database URLs/JWTs
- support custom redaction rules
- flag instruction-like repository content as possible prompt injection while preserving provenance
- bind HTTP to loopback by default
- require explicit external binding
- add authentication, restricted CORS, request limits, and rate limiting before treating HTTP as a remote API
- hide internal traces by default

# Milestone 9 — Optional SaaS control plane

Do not make SaaS a near-term dependency of the engine.

A future control plane may manage:

- users
- organizations
- memberships
- projects
- worker pairing
- API keys
- policy
- GitHub integration
- audit events
- usage
- billing

A private worker should continue to own:

- repository source
- indexing
- embeddings
- symbol/dependency graphs
- retrieval
- context generation

Hosted/VPC workers and direct-storage acceleration come later if product evidence justifies them.

# Performance work after candidate pruning

Do not prioritize these ahead of selective candidate indexing unless profiling says otherwise:

## Async/double buffering

The current loop is synchronous even though the pool has multiple reusable buffers. After candidate pruning, profile CUDA streams/events and overlap chunk N verification with chunk N+1 read/transfer.

## KvikIO / cuFile / GDS

The storage seam already exists. A future direct backend should populate the destination device allocation and return `device_ready=True` without changing candidate selection, verification, result mapping, MCP, or HTTP contracts.

Direct storage is an optimization, not the current product milestone.

# CI and documentation target

CI should continue expanding toward:

- Windows, Linux, macOS where practical
- supported Python versions
- unit/integration/MCP/HTTP/security/package/CPU smoke coverage
- Ruff and packaging checks
- retrieval quality and output-budget gates
- selected resource/performance gates only after stable baselines exist

Documentation should stay consistent with the current out-of-core architecture. In particular, do not repeat historical claims that exact search permanently stores two full corpus copies in VRAM or performs zero disk I/O after startup.

# Definition of done for the next major development cycle

The project should demonstrate all of the following:

1. Exact search scales beyond available VRAM with bounded memory.
2. Selective exact queries inspect/read a small fraction of the packed corpus.
3. `plan_change` or its successor returns compact structured agent context.
4. A reproducible C#/.NET agent benchmark compares coding-agent behavior with and without gpu-search-mcp.
5. The benchmark can report context/file/tool-call efficiency and correctness evidence.
6. CPU remains a supported correctness baseline; CUDA/MPS remain accelerators.
7. Documentation and benchmarks accurately describe the implemented architecture.

# Progress log

- **2026-07-20:** Added the backward-compatible unified search contract with intent-aware routing, structured primary/related results, dependency/test expansion, readiness metadata, warnings, and root filtering.
- **2026-07-21:** Added OpenAPI updates, read-only doctor, packaged setup for Claude/Codex, configuration backups/idempotence, and Python-only direction.
- **2026-07-22:** Completed C# symbol intelligence and deterministic agent change planning; added retrieval-quality manifests, baselines, and CI quality/output-budget gates.
- **2026-07-23:** Added content-addressed cache identities and crash-safe cache transactions with locks, stale recovery, rollback, and failure-injection coverage.
- **2026-08-08:** Refactored exact search to a packed out-of-core corpus with replaceable storage, bounded reusable GPU buffers, storage-agnostic verification, candidate-selection seam, and out-of-core metrics.
- **2026-08-08:** Added CUDA out-of-core baselines and resident-vs-out-of-core comparison. The 64 MiB baseline used 4 MiB reusable-buffer VRAM versus ~128 MiB in the former resident implementation, while dense all-chunk queries were 28–46% slower.
- **2026-08-08:** Expanded exact-search equivalence coverage across file/mmap/memory storage backends, tiny chunk sizes, UTF-8/NUL data, overlap, long queries, missing results, and `max_files`; validation reported 267 passing tests.

# Immediate queue

1. **NOW — Candidate pruning:** design and implement a selective candidate chunk index behind `CandidateSelector`; benchmark candidate percentage, physical-read ratio, index size/build cost, and result parity.
2. **NEXT — Agent evaluation:** build the first reproducible C#/.NET coding-agent task suite comparing the agent alone vs agent + gpu-search-mcp.
3. **NEXT — Context surface:** use benchmark failures to decide whether to extend `plan_change` or introduce a stable `prepare_context`-style API.
4. **THEN — C# quality:** improve structural intelligence only where agent-evaluation evidence shows meaningful gaps.
5. **LATER — pipeline/direct storage:** profile double buffering, then optionally KvikIO/cuFile/GDS after candidate pruning has reduced the dominant work.
6. **LATER — worker/SaaS:** build a persistent multi-repo private worker before any SaaS control plane.
