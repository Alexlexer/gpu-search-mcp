# gpu-search-mcp

A **local-first context engine for coding agents**.

GPU Search retrieves, combines, ranks, and compresses repository evidence so agents such as Codex and Claude can inspect less irrelevant code, use fewer context tokens, and reach the right implementation faster.

GPU acceleration is an implementation advantage, not a requirement: exact search works on CUDA, Apple MPS, and CPU.

## What works today

- Out-of-core exact search over a packed repository corpus.
- Bounded reusable buffers, so exact-search VRAM does not scale with repository size.
- Conservative trigram candidate pruning with a persistent checksummed index.
- CUDA, MPS, and CPU verification.
- Semantic search with a persistent embedding cache.
- Dependency and Git context.
- C#/.NET symbol intelligence: symbols, references, implementations, callers, callees, tests, ASP.NET/DI heuristics, confidence, and provenance.
- Deterministic token-budgeted `plan_change` context bundles.
- MCP stdio and local HTTP APIs.
- Local caches, secret redaction, indexed-root validation, diagnostics, and CI quality gates.
- An opt-in coding-agent A/B harness with a real Codex CLI adapter, validation-gated success, sanitized trajectories, and a five-task .NET suite.

The runtime is Python-only. The abandoned Rust rewrite is not part of the active architecture.

## Why

Coding agents often spend large amounts of context and tool calls discovering a repository before making a change.

GPU Search is being built around a different workflow:

```text
coding task
    |
    v
retrieval + structure + Git evidence
    |
    v
rank / deduplicate / token-budget
    |
    v
compact context bundle
    |
    v
coding agent
```

The product goal is not "faster grep". The goal is **less repository exploration per successfully solved task** while preserving or improving correctness.

## Architecture

```text
Repository
   |
   +--> packed corpus --> candidate index --> StorageBackend
   |                                      --> CPU / CUDA / MPS exact verification
   |
   +--> semantic index
   +--> dependency graph
   +--> symbol graph
   +--> Git state
             |
             v
        evidence fusion
             |
             v
      plan_change today
      prepare_context next
             |
             v
       Codex / Claude
```

Exact verification remains authoritative. Candidate indexes may produce false positives, but must not produce false negatives.

Source code and derived indexes remain local by default.

## Install

```bash
pipx install gpu-search-mcp
# or
uv tool install gpu-search-mcp
```

Development checkout:

```bash
git clone https://github.com/Alexlexer/gpu-search-mcp.git
cd gpu-search-mcp
python -m venv .venv
python -m pip install -e ".[test,all]"
```

## Run

```bash
gpu-search-mcp --directory /path/to/repo
```

Configure Codex or Claude:

```bash
gpu-search-mcp setup --client codex --yes
gpu-search-mcp setup --client claude --yes
```

Diagnostics:

```bash
gpu-search-mcp doctor
gpu-search-mcp doctor --json
```

HTTP mode is local-only by default:

```bash
gpu-search-mcp --directory /path/to/repo --http
```

Do not expose the HTTP API directly to the public internet.

## Main agent surfaces

- `search_code` — exact, semantic, hybrid, and symbol-oriented retrieval.
- `find_symbol`, `find_callers`, `find_callees`, `find_references`, `find_implementations`, `find_tests` — structural queries.
- `dep_impact` / `dep_imports` — dependency evidence.
- `gpu_read_block` / `gpu_skeleton` — targeted source expansion.
- `plan_change` — current high-level token-budgeted change context.

Lower-level search tools remain available for precise agent control.

## Current development direction

The near-term roadmap is deliberately evidence-driven:

1. **Agent A/B evaluation** — measure Codex alone vs Codex + GPU Search: task success, patch correctness, files inspected, tool calls, tokens, and time to relevant code.
2. **`prepare_context`** — turn retrieval, symbols, dependencies, tests, Git state, risks, and unknowns into one compact agent-facing context operation.
3. **Context quality** — ranking, deduplication, symbol-level snippets, and explicit token-budget allocation.
4. **Large-repository proof** — benchmark 1/10/30/100 GiB logical corpora and measure physical bytes read, RAM/VRAM, startup, and latency.
5. **Smarter candidate selection** — compare all/first-trigram/rarest/intersection strategies with zero false negatives.
6. **Adaptive CPU/GPU execution** — use measurements to choose the faster verifier for each candidate workload.
7. **Pluggable structural intelligence** — introduce a broad `StructureProvider` boundary so structure may come from built-in C#, Tree-sitter, LSP/compiler integrations, or external graph engines.

After that, benchmark failures decide the roadmap. Broad language count, Roslyn, GDS, multi-repo workers, and any SaaS/control plane are later work only when evidence justifies them.

See [`docs/agent-evaluation.md`](docs/agent-evaluation.md) for the opt-in Codex A/B workflow, [`docs/project-state.md`](docs/project-state.md) for the current implementation snapshot, and [`ROADMAP.md`](ROADMAP.md) for the development sequence.

## Principles

- Local/private by default.
- CPU correctness is mandatory; GPU is optional acceleration.
- Benchmark before optimizing.
- Measure agent outcomes, not just search latency.
- Keep exact verification authoritative.
- Preserve MCP/HTTP compatibility where practical.
- Prefer small, reviewable changes.
- Do not claim token savings or GPU superiority without comparable measurements.

## License

MIT — see [`LICENSE`](LICENSE).
