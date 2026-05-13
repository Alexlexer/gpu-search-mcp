# gpu-search-mcp `v0.1.0`

A GPU-accelerated codebase search server built as an [MCP](https://modelcontextprotocol.io/) tool. It loads your source files directly into RTX VRAM and runs searches as vectorized CUDA operations via PyTorch — no custom kernels, no native extensions.

> **Status:** Working prototype, used daily on a single machine. Core search is solid; some features described below are best-effort (see [Limitations](#known-limitations)).

## How it works

On startup the server prepares three search-time data structures:

1. **Pattern index** — every indexed source file is read into VRAM as `uint8` tensors. Exact queries use a first-char GPU filter and vectorized window checks.
2. **Dependency graph** — project imports are parsed with regex heuristics into a sparse graph so the agent can answer "what imports this file?" before editing. This is **best-effort** and not compiler-accurate (see [dep graph limitations](#dependency-graph)).
3. **Semantic cache loader** — the embedding model is warmed and any on-disk semantic caches are merged into memory. If no cache exists yet, run `gpu_semantic_index` once to build it.

A `watchdog` watcher keeps the pattern, semantic, and dependency indexes in sync as files change. Search results are also re-ranked using recent git activity and file mtimes so actively edited files surface first.

## Security behavior

- **`.env` files are excluded from indexing by default.** To opt in, pass `--allow-env-files` to the server (use only on non-sensitive repos).
- **Search output is redacted** — common secret patterns (API keys, bearer tokens, passwords, connection strings, PEM private keys) are replaced with `[REDACTED]` before being returned to the LLM. This is best-effort pattern matching, not a DLP scanner.
- If you index a `.env` file via `--allow-env-files`, the redaction layer still applies to search output, but the raw bytes live in VRAM for the lifetime of the server process.

## Requirements

- Python 3.10+
- GPU recommended but not required — the server auto-selects the best available backend:
  - **NVIDIA (CUDA)** — full acceleration, tested on RTX 4060
  - **Apple Silicon (MPS)** — Metal GPU acceleration on M1/M2/M3/M4
  - **CPU** — works everywhere, slower for large codebases

## Installation

The simplest path is to run the installer with Python 3.10+. If `uv` is on `PATH`, the installer prefers it automatically:

```bash
python3.12 install.py
```

It creates a local `.venv`, installs dependencies, and registers the MCP server for Claude Code and Codex.

### Installer flags

```
--dry-run           Print what would be changed without writing any files
--yes               Skip the directory prompt; use the current directory
--no-claude         Skip Claude Code registration
--no-codex          Skip Codex registration
--backup-configs    Create .bak copies before overwriting config files (on by default)
--installer uv|pip  Force a specific package backend
```

You can force a backend explicitly:

```bash
python3.12 install.py --installer uv
python3.12 install.py --installer pip
```

### Using uv directly

This repo includes a `pyproject.toml` so `uv` can manage the environment directly:

```bash
uv venv --python 3.12
uv sync
```

## Usage

### Run the MCP server

```bash
# Index the current directory on startup
.venv/bin/python gpu_service/mcp_server.py

# Or use the CLI entrypoint (after pip install -e . or uv sync)
gpu-search-mcp --directory /absolute/path/to/your/project

# Allow .env indexing (opt-in, prints a warning)
gpu-search-mcp --directory /path/to/project --allow-env-files
```

### Smoke test

```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --with-semantic   # also attempts model loading
```

### MCP tools

**Use `search_code` for everything** — it auto-routes to the right backend:

| Query type | Example | Routes to |
|---|---|---|
| Identifier / symbol / literal | `"handleError"`, `"AUTH_TOKEN"` | Pattern search |
| Natural language | `"where is error handling middleware"` | Semantic search |
| Mixed intent / exploration | `"auth token refresh"`, `mode="hybrid"` | Pattern + semantic in parallel |

```python
search_code("handleError")                                   # exact match
search_code("where is user authentication handled")          # semantic match
search_code("auth token refresh", mode="hybrid", top_k=5)   # combined ranking
```

**Low-level tools:**

| Tool | Description |
|------|-------------|
| `gpu_search(query, case_sensitive?)` | Exact-text pattern search. Use when `case_sensitive` matters. |
| `gpu_semantic_search(query, top_k?)` | Meaning-based search. Returns scored chunks with file + line range. |
| `gpu_index(directory)` | Rebuild pattern index (e.g. after large refactor). |
| `gpu_semantic_index(directory, append?, force?)` | Build or rebuild the semantic embedding cache. |
| `dep_index(directory)` | Build the import dependency graph. |
| `dep_impact(filepath)` | Show all files transitively affected by changes to a file. |
| `dep_imports(filepath)` | Show the direct project imports of a file. |
| `gpu_add_directory(directory)` | Add a directory to the permanent startup config. |
| `gpu_update_file(filepath)` | Re-index one file after editing. |
| `gpu_read_block(filepath, line)` | Expand a search hit to its enclosing function/class block. |
| `gpu_skeleton(filepath, match_lines?)` | Show a folded file outline with matched blocks expanded. |
| `gpu_stats()` | Show index status, VRAM usage, and background progress. |

### Zero-overhead sessions

The server reads `~/.gpu-search-config.json` on startup and auto-indexes every listed directory. The installer writes to this file automatically. You can also add directories at runtime:

```
gpu_add_directory("/path/to/project")
```

## File types indexed

`.py .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .java .cs .rb .php .swift .kt .json .yaml .yml .toml .md .txt .html .css .scss .sql .sh .bat .ps1 .cfg .ini .xml`

`.env` is excluded by default. Pass `--allow-env-files` to opt in.

Directories skipped: `.git node_modules __pycache__ .venv venv dist build .next .nuxt target bin obj .idea .vscode .mypy_cache`

## Benchmark

> **Example results** — collected by the author on a specific machine. Your results will depend on hardware, OS, and repo size. Run `benchmarks/run_benchmark.py` to measure on your own setup.

Tested against the [VS Code](https://github.com/microsoft/vscode) repo — 12,259 files, 285 MB of source. Measured as direct Python calls (no MCP transport overhead). Hardware: RTX 4060 (8 GB VRAM), Windows 11.

### Pattern search vs ripgrep (example results)

| Query | Matches | gpu-search | ripgrep warm | ripgrep cold |
|-------|---------|-----------|--------------|--------------|
| `ICodeEditor` | 428 files | **10ms** | ~110ms | ~6,300ms |
| `createTextModel` | 95 files | **8ms** | ~135ms | ~790ms |
| `disposeOnReturn` | 3 files | **7ms** | ~200ms | ~200ms |
| `handleError` | 14 files | **8ms** | ~120ms | ~400ms |
| `addEventListener` | 109 files | **10ms** | ~115ms | ~500ms |

- **10–15× faster than ripgrep (warm)** — searches run entirely in VRAM (272 GB/s), zero disk I/O after startup.
- Cold start: ripgrep reads from disk (0.4–6s per query); gpu-search indexes once at startup (~3s), then every search is sub-15ms.

See [benchmarks/methodology.md](benchmarks/methodology.md) for the full measurement methodology.

### Semantic search (example results)

Semantic search finds code by meaning — no exact match needed. Runs as a single GPU matmul over 93,635 embedded chunks.

| Query | gpu-search |
|-------|-----------|
| `"where is undo redo handled"` | **20ms** |
| `"how does syntax highlighting work"` | **13ms** |
| `"error handling in file system"` | **10ms** |
| `"authentication and login flow"` | **10ms** |

Semantic index: 93,635 chunks × 384 dims = 137 MB VRAM. Built once (~2 min on GPU), then loads from disk cache in ~3s on every restart.

## Architecture

```
gpu_service/
├── gpu_index.py            # GpuFileIndex — VRAM byte loading and vectorized pattern search
├── gpu_semantic_index.py   # SemanticIndex — chunking, embedding, disk cache, cosine search
├── gpu_dep_index.py        # DepIndex — sparse import graph + blast radius analysis
├── ast_expand.py           # Tree-sitter block expansion and skeleton mode
├── git_state.py            # Recency weighting from git diff/commit history
├── redact.py               # Secret redaction for search output
└── mcp_server.py           # FastMCP server — tool surface, routing, watchers, startup flow
```

## Known limitations

### Dependency graph

The dependency graph is built with **regex/heuristic import parsing**, not a full compiler. It handles common patterns in Python, JS/TS, Go, Rust, Java, C#, and Ruby, but will miss:
- Dynamic imports (`importlib`, `require()` with variables)
- Conditional imports
- Generated or bundled code

Use `dep_impact` results as a starting point, not a guarantee.

### Token usage

`search_code` prefers AST-expanded blocks over raw line windows, which improves context quality but can be expensive on large repos. A single call can cost hundreds to a few thousand tokens.

Workarounds:
- Use `top_k` to limit results: `search_code("query", top_k=3)`
- Use `gpu_search` for exact identifiers (pattern results are shorter)
- Use `mode="pattern"` when you do not want semantic expansion
- Avoid `dep_impact` on highly-imported files (core utilities can list hundreds of dependents)

### Pattern + dependency indexes are rebuilt on every restart

The pattern and dependency indexes are rebuilt from disk on every server restart. Semantic embeddings are cached to disk and load in seconds; exact-text and dependency state are still reconstructed at launch (~3–10s for a large repo).

### Tree-sitter coverage is narrow

AST expansion and skeleton mode currently target Python and TypeScript/JavaScript. Unsupported file types fall back to line-window snippets.

### Secret redaction is best-effort

The redaction layer catches common patterns but is not a comprehensive DLP scanner. Do not rely on it as your only secret protection — keep secrets out of source files in the first place.

## Troubleshooting

### No GPU / CUDA not detected

The server falls back to CPU automatically. Pattern search still works; semantic embedding will be slower. Check `gpu_stats()` to see which device is active.

### Semantic search unavailable

If `gpu_stats()` shows `semantic: not built`, run `gpu_semantic_index /path/to/project` once. This downloads the `BAAI/bge-small-en-v1.5` model (~90 MB) on first use — requires internet access or a pre-cached HuggingFace download.

### CUDA detected but model won't load

The model is loaded with `sentence-transformers`. If you see a HuggingFace network error, the model isn't cached locally. Ensure internet access the first time, or manually download the model and set `HF_HUB_OFFLINE=1`.

### Stale semantic cache rebuilt

If the server logs `Cache stale: ... changed — rebuilding`, the directory contents changed since the cache was built (new files, modified files, or chunking parameters changed). The cache is deleted and rebuilt automatically on the next `gpu_semantic_index` call.

### MCP server not appearing in Claude

1. Check `~/.claude.json` contains an `mcpServers.gpu-search` entry pointing to the correct Python interpreter.
2. Restart Claude Code after changing MCP config.
3. Run `python scripts/smoke_test.py` to verify the server works without MCP transport.

## When not to use this

- **Small repos (< 500 files):** ripgrep is fast enough and has zero startup cost. gpu-search's amortized advantage only appears when you run many searches per session.
- **Environments without persistent processes:** The startup index build takes ~3–10s. If your agent restarts frequently, that cost adds up.
- **VRAM-constrained machines:** The pattern index for a 285 MB repo uses ~570 MB VRAM (two uint8 corpus copies). If your GPU is already heavily loaded, use CPU mode or a smaller repo.
- **When you need compiler-accurate dependency analysis:** The dep graph is regex-based. Use a language server or proper static analysis tool if accuracy matters.
- **When secret leakage is a critical risk:** The redaction layer helps but is not foolproof. Do not index repos containing production secrets.

## License

MIT
