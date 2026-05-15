# Changelog

## 0.1.0 — Unreleased

### Added

- GPU pattern search over a whole repo as a single VRAM byte corpus (272 GB/s VRAM throughput via PyTorch).
- Semantic search with persistent embedding cache (BAAI/bge-small-en-v1.5, 384 dims, GPU cosine similarity).
- Dependency graph: transitive import impact analysis for Python, JS/TS, Go, Rust, Java, C#, Ruby.
- C# AST support: tree-sitter block expansion, brace-matching fallback, `using`/namespace/base-type dependency heuristics.
- HTTP mode (`--http`) for local integrations such as LegacyLens.
- Structured HTTP responses on all search and read endpoints: `results`, `file`, `absoluteFile`, `lineStart`, `lineEnd`, `content`, `language`, `impactedFiles`.
- Root/path validation on all HTTP file endpoints — path traversal and outside-root reads return 400.
- Benchmark CLI (`gpu-search-bench`) with ripgrep comparison, latency percentiles, and JSON output.
- Secret redaction on all search output (API keys, bearer tokens, passwords, connection strings, PEM keys, AWS keys).
- Persistent pattern and dependency cache under `.gpu-search-cache/` for fast restarts.
- Watchdog-based live re-indexing on file save (debounced, 2 s window).
- LegacyLens integration support: stable structured JSON surface alongside backward-compatible `result` strings.
- `_infer_language` helper: maps file extension to language identifier for structured read responses.

### Changed

- HTTP read endpoints (`/read/block`, `/read/skeleton`, `/dependency/impact`) now return structured JSON fields in addition to the existing `result` string.
- CI dependency installation consolidated from `pip install -r requirements.txt && pip install pytest` to `pip install -e ".[test,ast]"`, eliminating drift between `requirements.txt` and `pyproject.toml`.

### Security

- HTTP mode binds to `127.0.0.1` by default; `0.0.0.0` requires `--host 0.0.0.0` and logs a warning.
- All HTTP file endpoints validate paths against configured/indexed roots. Path traversal (`../`) and absolute paths outside roots are rejected with HTTP 400.
- `.env` files are excluded from indexing by default; opt-in requires `--allow-env-files`, which prints a warning.
- Search output is redacted before being returned to any caller (best-effort pattern matching, not DLP-level).

### Documentation

- Expanded HTTP mode section in README with full endpoint reference, per-endpoint request/response examples, 400 error example, security/binding guidance, and LegacyLens integration notes.
- Added `CHANGELOG.md`.
- Added `docs/release-checklist.md`.
- Added `docs/releases/v0.1.0.md` release notes.

### Testing

- HTTP path safety tests: empty filepath rejection, normalized-path acceptance, skeleton endpoint outside-root rejection, parent-traversal rejection, absolute-path-outside-root rejection.
- Structured search response schema tests: required fields on all result types, snippet redaction, compact snippet length limit, hybrid deduplication, empty result shape.
- Structured read response tests: `/read/block`, `/read/skeleton`, `/dependency/impact` structured field coverage.
- Added `pytest-timeout>=2.3.0` to `[test]` extras to support the CI `--timeout=60` flag without separate installation.
