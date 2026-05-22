# Rust Rewrite Tracking Plan

Status: planning / tracking only  
Created: 2026-05-22  
Repo: `Alexlexer/gpu-search-mcp`

This document is a checkpoint for a possible Rust rewrite of `gpu-search-mcp`. It is intentionally not a commitment to replace the current Python implementation immediately.

## Decision

Do **not** rename or split the repository yet.

Keep the current repository:

```text
Alexlexer/gpu-search-mcp
```

Add Rust incrementally alongside the existing Python implementation.

Recommended layout:

```text
gpu-search-mcp/
  gpu_service/             # current Python implementation
  crates/
    gpu-search-core/       # Rust core library
    gpu-search-http/       # future Rust HTTP server
    gpu-search-mcp/        # future Rust MCP server
  Cargo.toml               # Rust workspace root
```

## Why not a new repo yet?

Keeping the Rust rewrite in this repo is better for now because:

- Existing GitHub history, docs, issues, and releases stay intact.
- Python implementation remains stable while Rust catches up.
- Rust and Python behavior can be compared in the same repo.
- CI can test both implementations.
- Migration can happen gradually without breaking current users.
- Portfolio story is stronger: incremental Rust rewrite of hot paths in a real tool.

A separate repo such as `gpu-search-rs` can be considered later if the Rust version becomes a standalone product.

## High-level goal

Turn `gpu-search-mcp` into a fast, portable local code intelligence engine, with Rust handling deterministic/indexing-heavy work and Python remaining available for sentence-transformers semantic embeddings until a better Rust embedding backend is chosen.

## Expected performance impact

Rust should help most with:

| Area | Expected impact |
|---|---|
| File walking | 2-5x faster |
| Cache load/save | 2-4x faster |
| Dependency graph parsing | 2-10x faster depending parser strategy |
| CPU exact search | 5-20x faster with `memchr` / `aho-corasick` / SIMD |
| HTTP overhead | Lower latency and memory use |
| Startup/warm cache | 2-10x faster possible |

Rust will **not automatically improve semantic search**, because semantic search currently depends on Python `sentence-transformers`.

Semantic options later:

1. Keep Python sentence-transformers as a sidecar.
2. Use ONNX Runtime from Rust.
3. Use Candle from Rust.
4. Keep semantic optional and focus Rust on pattern/dependency/search infrastructure.

Recommended first semantic approach: keep Python sidecar.

## Rewrite principles

- No big-bang rewrite.
- Keep Python version working throughout.
- Keep HTTP/MCP response shapes compatible.
- Keep exact/pattern search working without semantic model.
- Do not add Ollama to `gpu-search-mcp`.
- Ollama belongs in LegacyLens for LLM review/audit summaries.
- Do not remove current Python code until Rust has parity.
- Prefer measurable milestones and benchmarks.

## Phased plan

### Phase 0 — Tracking and working protocol

- [x] Create this planning document.
- [x] Move the plan into source control now that the Rust rewrite is active.
- [x] Decide first Rust implementation branch name: `feat/rust-core-prototype`.
- [x] Keep this document updated as each milestone lands.

Working protocol:

- Use small, focused PRs.
- Prefer PRs that can be reviewed and merged independently.
- Enable auto-merge after local checks and CI pass.
- Do not let large rewrite branches accumulate unrelated changes.
- Update this document in the same PR when a checklist item is completed.
- Keep Python behavior stable while Rust code is added alongside it.

Suggested branch:

```text
feat/rust-core-prototype
```

### Phase 1 — Rust workspace scaffold

Goal: establish Rust project structure without changing existing behavior.

Tasks:

- [x] Add root `Cargo.toml` workspace.
- [x] Add `crates/gpu-search-core`.
- [x] Add basic library exports.
- [x] Add simple unit test.
- [x] Add README note that Rust core is experimental.
- [x] Ensure Python tests still pass.
- [ ] Add Rust checks to CI only if stable and quick.

Suggested crates:

```text
crates/
  gpu-search-core/
    Cargo.toml
    src/lib.rs
    src/file_discovery.rs
    src/pattern.rs
    src/line_index.rs
    src/cache.rs
    src/deps.rs
```

Recommended dependencies:

```toml
ignore = "0.4"
memchr = "2"
aho-corasick = "1"
rayon = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
anyhow = "1"
```

### Phase 2 — File discovery

Goal: Rust discovers indexed files the same way Python does.

Tasks:

- [x] Port indexed extension list.
- [x] Port skip directory list.
- [x] Exclude `.env` by default.
- [x] Add `allow_env_files` option.
- [x] Respect max file size.
- [x] Return stable sorted file list.
- [x] Add tests for dotfiles, skipped dirs, binary/large files.

Output DTO:

```rust
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub size: u64,
    pub modified_ns: u128,
}
```

### Phase 3 — CPU pattern-search prototype

Goal: first real Rust search path.

Tasks:

- [x] Load discovered files into an in-memory corpus or per-file byte buffers.
- [x] Implement case-sensitive and case-insensitive exact search.
- [x] Return line numbers and snippets.
- [x] Add compact result structs matching current HTTP DTO concepts.
- [ ] Add benchmarks against Python pattern search and ripgrep.

Recommended implementation options:

- `memchr` for single-token exact match.
- `aho-corasick` for multi-query or signal scan later.
- `rayon` for parallel per-file search.

Result DTO:

```rust
pub struct PatternMatch {
    pub file: PathBuf,
    pub line: usize,
    pub content: String,
}
```

### Phase 4 — Line offsets and snippets

Goal: fast line mapping and low-token output.

Tasks:

- [ ] Store newline offsets per file.
- [ ] Convert byte offsets to line numbers quickly.
- [ ] Return one-line snippets.
- [ ] Add compact/normal/full context modes later.

### Phase 5 — Cache metadata compatibility

Goal: Rust uses explicit cache metadata compatible in spirit with `.gpu-search-cache/cache-meta.json`.

Tasks:

- [ ] Implement cache schema constants.
- [ ] Read/write `cache-meta.json`.
- [ ] Compute lightweight source fingerprint.
- [ ] Safely invalidate stale/incompatible cache.
- [ ] Never touch source files.

Cache structure should stay under:

```text
.gpu-search-cache/
```

### Phase 6 — Dependency graph

Goal: Rust dependency analysis matches current heuristic behavior first.

Tasks:

- [ ] Port Python import parsing.
- [ ] Port JS/TS import parsing.
- [ ] Port C# using/namespace/type/base/interface heuristics.
- [ ] Preserve impact reasons.
- [ ] Return confidence/limitations/reasons.
- [ ] Add tests using small temp repos.

Do **not** add Roslyn in Rust core at this stage.

### Phase 7 — Tree-sitter support

Goal: improve C# and other language parsing.

Tasks:

- [ ] Add `tree-sitter`.
- [ ] Add `tree-sitter-c-sharp`.
- [ ] Parse C# class/interface/record/struct/enum/method/property/namespace/using.
- [ ] Keep regex fallback.
- [ ] Add tests for controller actions and inheritance/interface usage.

### Phase 8 — Rust HTTP server

Goal: expose current HTTP API shape from Rust.

Recommended stack:

- `tokio`
- `axum`
- `serde`
- `tower-http`
- `tracing`
- `utoipa` or `aide` for OpenAPI later

Endpoints to preserve:

```text
GET  /health
GET  /stats
GET  /diagnostics
GET  /semantic/model/status
POST /search/code
POST /search/hybrid
POST /search/semantic
POST /read/block
POST /read/skeleton
POST /dependency/impact
POST /scan/signals
```

Initial Rust HTTP server can expose only:

```text
GET  /health
GET  /stats
GET  /diagnostics
POST /search/code
```

Then expand.

### Phase 9 — MCP server

Goal: Rust MCP stdio compatibility.

Options:

1. Use a Rust MCP SDK if stable enough.
2. Implement JSON-RPC stdio directly.
3. Keep Python MCP as wrapper over Rust core until Rust HTTP/core is stable.

Recommended: keep Python MCP wrapper first.

### Phase 10 — Semantic search strategy

Goal: avoid blocking Rust rewrite on ML stack.

Short term:

- Keep Python sentence-transformers semantic index.
- Rust reports semantic unavailable/sidecar status.
- Rust pattern/dependency/search work independently.

Later options:

- ONNX Runtime embeddings.
- Candle embeddings.
- Python sidecar via local HTTP or stdio.

Do not add Ollama to `gpu-search-mcp`.

## First implementation milestone

Branch:

```text
feat/rust-core-prototype
```

Scope:

- Add Rust workspace.
- Add `gpu-search-core` crate.
- Implement file discovery.
- Implement CPU exact pattern search.
- Add unit tests.
- Add a small benchmark or example CLI if simple.
- Do not replace Python paths yet.

Acceptance:

- Python tests still pass.
- Rust tests pass.
- No endpoint behavior changes.
- No response shape changes.
- No semantic model changes.

## Future repo decision

Keep this repo unless Rust becomes independently useful enough to split.

Possible future split:

```text
Alexlexer/gpu-search-rs
```

Only consider this after:

- Rust core has pattern search parity.
- Rust dependency graph has C# support.
- Rust HTTP API can serve basic endpoints.
- Benchmarks show clear value.

## Notes

This document is the live tracking point for the Rust rewrite. Update it as PRs land so the migration stays visible and incremental.

## Progress log

- 2026-05-22: Started `feat/rust-core-prototype` as the first implementation PR. Added an experimental Cargo workspace and `gpu-search-core` crate with basic indexable-file helpers and unit tests. No Python runtime behavior changes.
- 2026-05-22: Started `feat/rust-file-discovery` as the second implementation PR. Added Rust file discovery with default skip dirs, `.env` safety, max-size and binary-file filtering, stable sorted output, and unit tests. No Python runtime behavior changes.
- 2026-05-22: Started `feat/rust-pattern-search` as the third implementation PR. Added dependency-free CPU exact pattern search with case-sensitive/case-insensitive matching, line numbers, snippets, global result limits, and unit tests. No Python runtime behavior changes.

