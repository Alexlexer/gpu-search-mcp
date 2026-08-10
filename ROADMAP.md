# gpu-search-mcp roadmap

Status: active  
Direction: Python-only, local-first, evidence-driven.

## Product target

Build a **local context data plane for coding agents**: retrieve, combine, rank, and compress the minimum repository evidence needed to solve a task.

Success is measured primarily by:

- task/test success and patch correctness
- input/context tokens per successful task
- irrelevant files inspected
- tool calls
- time to relevant implementation
- physical repository I/O at large scale

GPU acceleration matters only when measurements show it improves the workload. CPU correctness is mandatory.

## Baseline to preserve

- out-of-core packed exact-search corpus
- persistent conservative trigram candidate index
- bounded CPU/GPU buffers
- CUDA/MPS/CPU exact verification
- semantic retrieval
- dependency and Git evidence
- C# symbol/relationship intelligence
- deterministic token-budgeted `plan_change`
- MCP and local HTTP transports
- local caches, diagnostics, redaction, and root isolation

## Development sequence

### 1. Agent evaluation — NOW

Build a reproducible A/B harness:

```text
coding agent alone
vs
same agent + gpu-search-mcp
```

Start with realistic C#/.NET tasks. Record task success, tests, patch correctness, files inspected, tool calls, input/output tokens when exposed, GPU Search context size, and timing.

Full agent runs must be opt-in; normal CI tests only harness logic and deterministic fixtures.

**Exit gate:** reliable baseline data exists. No unmeasured token-saving claims.

### 2. `prepare_context`

Create one stable, token-budget-aware operation that reuses current `plan_change`/retrieval logic and returns the most useful implementation, structural, dependency, test, configuration, Git, risk, and unknown evidence.

Keep low-level tools available.

**Exit gate:** a normal coding task can begin with one compact context request instead of many exploratory searches.

### 3. Context quality

Use A/B trajectories to improve ranking, deduplication, symbol-level snippets, evidence confidence/provenance, and token-budget allocation.

**Exit gate:** benchmark shows lower context/exploration cost without lower task success.

### 4. Large-repository proof

Benchmark logical corpora around 1, 10, 30, and 100 GiB.

Measure candidate ratio, physical bytes read, index size/build/load time, cold/warm startup, RAM/VRAM, storage/H2D/kernel/mapping time, and p50/p95 latency.

**Exit gate:** quantify how much physical data a selective query touches on a 100 GiB corpus.

### 5. Smarter candidate selection

Compare:

- all chunks
- current/first trigram
- rarest trigram
- intersected trigrams

Exact verification remains authoritative and candidate filtering must have zero false negatives for supported semantics.

**Exit gate:** materially lower candidate/read ratio with measured index and CPU overhead.

### 6. Adaptive CPU/GPU verification

Measure crossover points rather than assuming GPU is always faster.

Potential policy:

```text
tiny candidate set   -> CPU
larger candidate set -> GPU
```

Use candidate bytes, device state, transfer cost, and measured latency.

**Exit gate:** adaptive policy matches exact results and beats a fixed backend across representative workloads.

### 7. Structural provider boundary

Generalize structural intelligence behind a broad `StructureProvider`-style contract.

A provider may represent one language, many languages, Tree-sitter, LSP/compiler analysis, or an external graph engine. The planner consumes normalized symbols/edges/capabilities rather than depending directly on C#.

First migrate existing C# behavior without rewriting it.

**Exit gate:** existing C# tests remain compatible through the generic provider boundary.

### 8. Second structural backend

Prove the provider design with one fundamentally different backend, such as Tree-sitter, LSP, or an external structural graph adapter.

Do not race to implement dozens of languages.

**Exit gate:** `prepare_context` works unchanged with two different structural sources.

## After step 8

Stop following a rigid feature list. Choose work from measured bottlenecks and agent failures.

Possible later work:

- deeper ASP.NET/DI/options/EF/MediatR intelligence
- Roslyn/compiler-backed precision where justified
- provider SDK
- compressed postings
- pinned buffers, CUDA streams, async transfer/verification
- KvikIO/cuFile/GDS when host staging is proven expensive
- persistent multi-repository worker
- authenticated/versioned remote API
- optional control plane/SaaS

## Current non-goals

- Rust rewrite
- broad language-count competition
- GPU-required correctness
- GDS before profiling
- Kubernetes/microservices
- mandatory cloud/backend services
- premature SaaS
- performance or token claims without reproducible evidence

## Definition of the next major success

The next development cycle succeeds when we can demonstrate:

1. whether GPU Search improves real coding-agent tasks;
2. how many context tokens/files/tool calls it saves or costs;
3. a stable `prepare_context` surface;
4. selective search over repositories far larger than VRAM;
5. measured CPU/GPU execution choices;
6. structural intelligence that is no longer architecturally tied to C#.
