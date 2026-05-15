# Changelog

## 0.1.0 — Unreleased

### Added

- GPU pattern search over a whole repo as a single VRAM byte corpus.
- Semantic search with persistent embedding cache (BAAI/bge-small-en-v1.5).
- Dependency graph: transitive import impact analysis for Python, JS/TS, Go, Rust, Java, C#, Ruby.
- C# AST support: tree-sitter block expansion, brace-matching fallback, namespace/base-type dependency heuristics.
- HTTP mode (`--http`) for local integrations such as LegacyLens.
- Structured HTTP responses for all search and read endpoints (`results`, `file`, `absoluteFile`, `lineStart`, `lineEnd`, `content`, `language`, `impactedFiles`).
- Root/path validation on all HTTP file endpoints — path traversal returns 400.
- Benchmark CLI (`gpu-search-bench`) with ripgrep comparison and JSON output.
- Secret redaction on all search output (API keys, bearer tokens, passwords, connection strings, PEM keys).
- Persistent pattern and dependency cache under `.gpu-search-cache/`.
- Watchdog-based live re-indexing on file save.
- CI quality gates: pytest, Ruff, CPU/no-GPU compatibility, smoke test.
