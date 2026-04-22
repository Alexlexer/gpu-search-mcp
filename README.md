# gpu-search-mcp `v0.1.0`

A GPU-accelerated codebase search server built as an [MCP](https://modelcontextprotocol.io/) tool. It loads your source files directly into RTX VRAM and runs searches as vectorized CUDA operations via PyTorch — no custom kernels, no native extensions.

## How it works

On startup the server prepares three search-time data structures:

1. **Pattern index** — every indexed source file is read into VRAM as `uint8` tensors. Exact queries use a first-char GPU filter and vectorized window checks.
2. **Dependency graph** — project imports are parsed into a sparse graph so the agent can answer "what imports this file?" before editing.
3. **Semantic cache loader** — the embedding model is warmed and any on-disk semantic caches are merged into memory. If no cache exists yet, run `gpu_semantic_index` once to build it.

A `watchdog` watcher keeps the pattern, semantic, and dependency indexes in sync as files change. Search results are also re-ranked using recent git activity and file mtimes so actively edited files surface first.

## Requirements

- Python 3.10+
- GPU recommended but not required — the server auto-selects the best available backend:
  - **NVIDIA (CUDA)** — full acceleration, tested on RTX 4060
  - **Apple Silicon (MPS)** — Metal GPU acceleration on M1/M2/M3/M4
  - **CPU** — works everywhere, slower for large codebases

## Installation

The simplest path is to run the installer with Python 3.10+:

```bash
python3.12 install.py
```

It creates a local `.venv`, installs dependencies into it, and registers the MCP server for Claude Code and Codex.

**NVIDIA (Windows/Linux):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mcp watchdog numpy sentence-transformers
```

**Apple Silicon (Mac):**
```bash
pip install torch torchvision
pip install mcp watchdog numpy sentence-transformers
```

## Usage

### Run the MCP server

```bash
# Index the current directory on startup
.venv/bin/python gpu_service/mcp_server.py

# Or specify a directory
.venv/bin/python gpu_service/mcp_server.py --directory /absolute/path/to/your/project
```

If you are not using a local virtualenv, replace `.venv/bin/python` with any Python 3.10+ interpreter that has the project dependencies installed.

On first run, call `gpu_semantic_index` once to build the embedding cache. Every restart after that loads the cache automatically in the background, so semantic search is ready within seconds.

### MCP tools

**Use `search_code` for everything** — it auto-routes to the right backend and can also run in hybrid mode:

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

**Low-level tools** (when you need explicit control):

| Tool | Description |
|------|-------------|
| `gpu_search(query, case_sensitive?)` | Exact-text pattern search. Use when `case_sensitive` matters. |
| `gpu_semantic_search(query, top_k?)` | Meaning-based search. Returns scored chunks with file + line range. |
| `gpu_index(directory)` | Rebuild pattern index (e.g. after large refactor). |
| `gpu_semantic_index(directory, append?, force?)` | Build or rebuild the semantic embedding cache. |
| `dep_index(directory)` | Build the import dependency graph. |
| `dep_impact(filepath)` | Show all files transitively affected by changes to a file. |
| `dep_imports(filepath)` | Show the direct project imports of a file. |
| `gpu_add_directory(directory)` | Add a directory to the permanent startup config — auto-indexed on every future launch. |
| `gpu_update_file(filepath)` | Re-index one file after editing. |
| `gpu_read_block(filepath, line)` | Expand a search hit to its enclosing function/class block. |
| `gpu_skeleton(filepath, match_lines?)` | Show a folded file outline with matched blocks expanded. |
| `gpu_stats()` | Show index status, VRAM usage, and background progress. |

### Zero-overhead sessions

The server reads `~/.gpu-search-config.json` on startup and auto-indexes every listed directory — no tool calls needed. The installer writes to this file automatically. You can also add directories at runtime:

```
gpu_add_directory("/path/to/project")
```

This indexes the directory immediately and saves it so future sessions start pre-indexed.

### Wire it into Claude Code

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "gpu-search": {
      "command": "/absolute/path/to/project/.venv/bin/python",
      "args": ["/absolute/path/to/project/gpu_service/mcp_server.py", "--directory", "/absolute/path/to/your/project"]
    }
  }
}
```

Claude will automatically call `search_code` — no manual tool selection needed.

### Wire it into Codex

Register the server with the local Codex CLI:

```bash
codex mcp add gpu-search -- /absolute/path/to/project/.venv/bin/python /absolute/path/to/project/gpu_service/mcp_server.py --directory /absolute/path/to/your/project
```

You can inspect the registration with:

```bash
codex mcp list
```

The included installer registers `gpu-search` in both Claude Code and Codex, preferring `.venv/bin/python` when a local virtualenv exists.

## File types indexed

`.py .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .java .cs .rb .php .swift .kt .json .yaml .yml .toml .md .txt .html .css .scss .sql .sh .bat .ps1 .cfg .ini .xml .env`

Directories skipped: `.git node_modules __pycache__ .venv venv dist build .next .nuxt target bin obj .idea .vscode .mypy_cache`

## Benchmark

Tested on the [VS Code](https://github.com/microsoft/vscode) repo — 12,259 files, 285 MB of source.
Measured as direct Python calls (no MCP transport overhead). Hardware: RTX 4060 (8 GB VRAM).

### Pattern search vs ripgrep

| Query | Matches | gpu-search | ripgrep warm | ripgrep cold |
|-------|---------|-----------|--------------|--------------|
| `ICodeEditor` | 428 files | **10ms** | ~110ms | ~6,300ms |
| `createTextModel` | 95 files | **8ms** | ~135ms | ~790ms |
| `disposeOnReturn` | 3 files | **7ms** | ~200ms | ~200ms |
| `handleError` | 14 files | **8ms** | ~120ms | ~400ms |
| `addEventListener` | 109 files | **10ms** | ~115ms | ~500ms |

- **10–15× faster than ripgrep (warm)** — searches run entirely in VRAM (272 GB/s), zero disk I/O after startup.
- ripgrep re-reads from OS file cache on every search; gpu-search reads from VRAM once, searches forever.
- Cold start: ripgrep reads from disk (0.4–6s per query); gpu-search indexes once at startup (~3s), then every search is sub-15ms.

### Semantic search (no ripgrep equivalent)

Semantic search finds code by meaning — no exact match needed. Runs as a single GPU matmul over 93,635 embedded chunks.

| Query | gpu-search |
|-------|-----------|
| `"where is undo redo handled"` | **20ms** |
| `"how does syntax highlighting work"` | **13ms** |
| `"error handling in file system"` | **10ms** |
| `"authentication and login flow"` | **10ms** |

Semantic index: 93,635 chunks × 384 dims = 137 MB VRAM. Built once (~2 min on GPU), then loads from disk cache in ~3s on every restart.

## Known limitations (v0.1.0)

### Token usage
`search_code` now prefers AST-expanded blocks over raw line windows, which improves context quality but can still be expensive on large repos. Depending on query breadth and result count, a single call can still cost **hundreds to a few thousand tokens**.

Workarounds for now:
- Use the `top_k` parameter to limit results: `search_code("query", top_k=3)`
- Use `gpu_search` for exact identifiers — pattern results are much shorter than semantic ones
- Use `mode="pattern"` when you do not want semantic expansion
- Avoid calling `dep_impact` on highly-imported files (core utilities, shared types) — they can list hundreds of transitive dependents

### Pattern + dependency indexes are in-memory only
The pattern and dependency indexes are rebuilt from disk on every server restart. Semantic embeddings are cached to disk, but exact-text and dependency state are still reconstructed at launch.

### Tree-sitter coverage is intentionally narrow
AST expansion and skeleton mode currently target Python and TypeScript/JavaScript first. Unsupported file types fall back to line-window snippets or plain file summaries.

## Architecture

```
gpu_service/
├── gpu_index.py            # GpuFileIndex — VRAM byte loading and vectorized pattern search
├── gpu_semantic_index.py   # SemanticIndex — chunking, embedding, disk cache, cosine search
├── gpu_dep_index.py        # DepIndex — sparse import graph + blast radius analysis
├── ast_expand.py           # Tree-sitter block expansion and skeleton mode
├── git_state.py            # Recency weighting from git diff/commit history
└── mcp_server.py           # FastMCP server — tool surface, routing, watchers, startup flow
```

## License

MIT
