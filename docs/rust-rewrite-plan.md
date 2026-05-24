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
- [x] Add Rust checks to CI only if stable and quick.

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
- [x] Add benchmarks against Python pattern search and ripgrep.

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

- [x] Store newline offsets per file.
- [x] Convert byte offsets to line numbers quickly.
- [x] Return one-line snippets.
- [x] Add compact/normal/full context modes later.

### Phase 5 — Cache metadata compatibility

Goal: Rust uses explicit cache metadata compatible in spirit with `.gpu-search-cache/cache-meta.json`.

Tasks:

- [x] Implement cache schema constants.
- [x] Read/write `cache-meta.json`.
- [x] Compute lightweight source fingerprint.
- [x] Safely invalidate stale/incompatible cache.
- [x] Never touch source files.

Cache structure should stay under:

```text
.gpu-search-cache/
```

### Phase 6 — Dependency graph

Goal: Rust dependency analysis matches current heuristic behavior first.

Tasks:

- [x] Port Python import parsing.
- [x] Port JS/TS import parsing.
- [x] Port C# using/namespace/type/base/interface heuristics.
- [x] Preserve impact reasons for Python import edges.
- [x] Return heuristic analysis mode and impact reasons.
- [x] Add tests using small temp repos for Python imports.

Do **not** add Roslyn in Rust core at this stage.

### Phase 7 — Tree-sitter support

Goal: improve C# and other language parsing.

Tasks:

- [x] Add `tree-sitter`.
- [x] Add `tree-sitter-c-sharp`.
- [x] Parse C# class/interface/record/struct/enum/method/property/namespace/using.
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

Progress:

- [x] Scaffold `crates/gpu-search-http`.
- [x] Add experimental `GET /health`.
- [x] Add `GET /stats`.
- [x] Add `GET /diagnostics`.
- [x] Add `POST /search/code`.
- [x] Wire `POST /search/code` to Rust pattern search when started with `--directory`.
- [x] Add experimental `POST /search/hybrid` and `POST /search/semantic` compatibility routes.
- [x] Add experimental `POST /dependency/impact`.
- [x] Add experimental `POST /read/block`.
- [x] Add experimental `POST /read/skeleton`.

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
- 2026-05-22: Started `feat/rust-line-offsets` as the fourth implementation PR. Added `LineIndex` for newline offsets, fast byte-offset to line/snippet mapping, and wired Rust pattern search to use it. No Python runtime behavior changes.
- 2026-05-23: Started `feat/rust-cache-metadata` as the fifth implementation PR. Added dependency-free Rust cache metadata helpers for schema constants, `cache-meta.json` read/write, source fingerprints, entry validation, and safe legacy/schema mismatch handling. No Python runtime behavior changes.
- 2026-05-23: Started `feat/rust-python-deps` as the sixth implementation PR. Added Rust heuristic dependency graph foundations with Python import parsing, reverse impact traversal, hop counts, and advisory import reasons. JS/TS and C# remain future PRs.
- 2026-05-23: Started `feat/rust-js-ts-deps` as the seventh implementation PR. Added Rust JS/TS import parsing for static imports, side-effect imports, `require(...)`, relative module resolution, `index` files, reverse impact traversal, and advisory import reasons.
- 2026-05-24: Started `feat/rust-csharp-deps` as the eighth implementation PR. Added Rust C# heuristic dependency parsing for namespaces, using statements, class/interface/record/struct/enum declarations, type references, base classes, interfaces, reverse impact traversal, and advisory reasons.
- 2026-05-24: Started `ci/rust-checks` as the ninth small PR. Added GitHub Actions Rust checks for `cargo fmt --check`, `cargo test`, and `cargo check` alongside existing Python CI.
- 2026-05-24: Started `feat/rust-tree-sitter-csharp` as the tenth small PR. Added Tree-sitter and `tree-sitter-c-sharp`, a lightweight C# AST summary helper, and parser smoke tests for using/namespace/type/method/property symbols. Existing heuristic dependency fallback remains unchanged.
- 2026-05-24: Started `feat/rust-context-modes` as the eleventh small PR. Added Rust pattern-search context modes for compact, normal, and full snippets while preserving normal as the default result shape.
- 2026-05-24: Started `feat/rust-pattern-benchmark` as the twelfth small PR. Added a Rust core pattern-search benchmark example with JSON output and wired the existing benchmark script to optionally compare Python gpu-search, ripgrep, and the Rust prototype.
- 2026-05-24: Started `feat/rust-http-scaffold` as the thirteenth small PR. Added an experimental `gpu-search-http` crate with an Axum router and local-first `/health` endpoint. It does not replace the Python HTTP/MCP runtime.
- 2026-05-24: Started `feat/rust-http-stats` as the fourteenth small PR. Added experimental `GET /stats` with version, capability, limitation, and zero-index status metadata without indexing repositories or changing Python behavior.
- 2026-05-24: Started `feat/rust-http-diagnostics` as the fifteenth small PR. Added experimental `GET /diagnostics` with static local setup/readiness metadata and no repo scans, model loads, or Python runtime behavior changes.
- 2026-05-24: Started `feat/rust-http-search-code-stub` as the sixteenth small PR. Added experimental `POST /search/code` request/response DTOs and a structured not-ready response while Rust indexing remains unwired.
- 2026-05-24: Started `feat/rust-http-pattern-search` as the seventeenth small PR. Wired experimental Rust HTTP state to discover files from `--directory` and return Rust core pattern results from `POST /search/code`.
- 2026-05-24: Started `feat/rust-http-dependency-impact` as the eighteenth small PR. Added experimental `POST /dependency/impact` backed by the Rust heuristic dependency graph with root validation, advisory reasons, and structured not-ready/error responses.
- 2026-05-24: Started `feat/rust-http-read-block` as the nineteenth small PR. Added experimental `POST /read/block` with indexed-root validation and bounded line-range reads for UTF-8 source files.
- 2026-05-24: Started `feat/rust-http-read-skeleton` as the twentieth small PR. Added experimental `POST /read/skeleton` with indexed-root validation, Tree-sitter C# symbols, and a simple fallback skeleton for non-C# files.
- 2026-05-24: Started `feat/rust-http-search-aliases` as the twenty-first small PR. Added experimental `/search/hybrid` and `/search/semantic` compatibility routes; hybrid returns Rust pattern results with a warning while semantic returns structured not-ready guidance.
