# Current project state

Last reconciled: 2026-08-11

## Product direction

gpu-search-mcp is a **local-first context engine for coding agents**.

The product goal is to retrieve, combine, rank, and compress the minimum high-confidence repository evidence an agent needs to solve a task. The primary outcome is less irrelevant repository exploration and lower context cost without reducing correctness.

GPU acceleration is optional. CPU correctness remains mandatory. Source code and derived indexes remain local/private by default.

The authoritative runtime is Python-only. The abandoned Rust rewrite is historical and must not be reintroduced without an explicit change in direction.

## Implemented today

### Exact retrieval

Exact search is out-of-core:

```text
repository -> packed corpus -> CandidateSelector -> StorageBackend
           -> bounded buffers -> CPU/CUDA/MPS exact verification -> results
```

Implemented properties:

- versioned `.gpusearch/corpus.bin`, `files.idx`, and `chunks.idx`
- bounded reusable buffers independent of repository size
- file, mmap, and in-memory storage backends
- chunk-boundary-safe matching with cross-file rejection
- case-sensitive and case-insensitive exact search
- conservative trigram candidate pruning
- persistent checksummed trigram postings in `trigrams.idx`
- cache validation/rebuild for stale or corrupt candidate indexes
- query/read/transfer/kernel/resource instrumentation

Recent selective benchmarks showed that trigram pruning can materially reduce physical reads; persistent postings also remove the need to rebuild candidate data on every warm start. These are implementation benchmarks, not yet product-level agent claims.

### Semantic, dependency, and Git evidence

- sentence-transformers semantic retrieval with persistent cache
- heuristic dependency graph and impact queries
- Git evidence/ranking support

### Structural intelligence

The core model exposes generic `Symbol` and `SymbolEdge` concepts, while the current extractor is C#/.NET-specific and heuristic.

Current C# coverage includes:

- symbols and signatures
- references
- callers/callees
- implementations/inheritance/overrides
- tests
- ASP.NET endpoint heuristics
- DI registration heuristics
- confidence and provenance

This is useful but not compiler-accurate. Broad language coverage is not currently the product priority.

### Agent context

`plan_change` is the current high-level context surface. It combines exact, semantic, symbol, dependency, test, configuration, and Git evidence into deterministic token-budgeted plans with reasons, risks, unknowns, omissions, likely change set, and inspection order.

This is the part of the current implementation closest to the long-term product direction.

### Agent evaluation

The opt-in evaluation harness now has explicit provider usage semantics, correctness eligibility, deterministic baseline/GPU grouping, a sanitized Codex CLI adapter, and five realistic C#/.NET tasks at an immutable fixture commit. Real benchmark success requires deterministic validation; agent completion telemetry and patch production are not treated as correctness proof.

A small Codex pilot was attempted on Windows, but the local nested Codex workspace sandbox was read-only because its sandbox helper was unavailable. Those runs produced no patches and are environment diagnostics, not A/B evidence. The 30-run comparison remains pending and no product claim is supported yet. See docs/benchmarks/agent-eval-codex-pilot-2026-08-11.md.

### Reliability

Current foundations include:

- retrieval-quality manifests and regression metrics
- returned-token measurements
- CPU/no-GPU compatibility tests
- package/smoke validation
- content-addressed cache identities
- repository locks and stale-lock recovery
- staged writes, fsync, atomic promotion, rollback, and recovery coverage
- local-only default HTTP binding, root validation, and secret redaction

## What is not proven yet

Do not make strong product claims about the following until measured:

- percentage of Codex/Claude tokens saved
- files/tool calls saved per successful task
- task-success improvement
- 100 GiB repository behavior
- GPU superiority for every candidate workload

Those measurements are now the development priority.

## Current priorities

1. **Run the first trustworthy Agent A/B evaluation** — the harness, Codex adapter, and five-task suite are ready; execute 30 isolated runs in an environment where nested Codex can write its workspace, then inspect failures before making product claims.
2. **`prepare_context`** — promote context generation into one stable agent-facing operation built on existing planning/evidence logic.
3. **Context quality** — ranking, deduplication, symbol-level snippets, and token-budget allocation driven by evaluation failures.
4. **Large-corpus benchmark** — establish 1/10/30/100 GiB scaling and physical-read/RAM/VRAM behavior.
5. **Candidate improvements** — compare first/rarest/intersected trigram strategies while preserving exact-result parity.
6. **Adaptive execution** — benchmark CPU vs GPU crossover and choose execution based on candidate workload rather than assuming GPU always wins.
7. **Structural provider boundary** — generalize C# intelligence behind a `StructureProvider`-style interface capable of consuming built-in, Tree-sitter, LSP/compiler, or external graph intelligence.

After these phases, measured agent failures and performance profiles should determine the backlog.

## Later, only if justified

- deeper .NET intelligence such as ASP.NET routing, options/config, EF Core, MediatR, or Roslyn-backed resolution
- second structural backend/provider
- provider SDK
- compressed postings and deeper GPU pipeline optimization
- KvikIO/cuFile/GDS
- persistent multi-repository worker
- remote/public API hardening
- optional control plane/SaaS

## Non-goals for the current cycle

- Rust rewrite
- broad language-count race
- Kubernetes/microservices
- mandatory cloud services
- premature Roslyn integration
- GDS before profiling
- GPU-required correctness

When documentation conflicts, prefer current code/tests, then this document, then older benchmark/release prose.
