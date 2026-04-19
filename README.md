# gpu-search-mcp

A GPU-accelerated codebase search server built as an [MCP](https://modelcontextprotocol.io/) tool. It loads your source files directly into RTX VRAM and runs searches as vectorized CUDA operations via PyTorch — no custom kernels, no native extensions.

## How it works

On startup the server builds two indexes:

1. **Pattern index** — every source file is read into VRAM as `uint8` tensors. Queries use a first-char GPU filter then a vectorized window check. Sub-millisecond for exact identifiers.
2. **Semantic index** — files are chunked into ~40-line windows and embedded with [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) (~130 MB). Embeddings are cached to disk so every restart after the first is instant. Queries are a single GPU matmul.

A `watchdog` watcher keeps the pattern index in sync as you edit. The embedding model runs on CPU so it loads in seconds, not minutes.

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

On first run, call `gpu_semantic_index` once to build the embedding cache (~15s). Every restart after loads the cache automatically in the background — semantic search is ready within seconds of startup.

### MCP tools

**Use `search_code` for everything** — it auto-routes to the right backend:

| Query type | Example | Routes to |
|---|---|---|
| Identifier / symbol / literal | `"handleError"`, `"AUTH_TOKEN"` | Pattern search (VRAM, sub-ms) |
| Natural language | `"where is error handling middleware"` | Semantic search (cosine similarity) |

```
search_code("handleError")                          # → exact match
search_code("where is user authentication handled") # → semantic match
```

**Low-level tools** (when you need explicit control):

| Tool | Description |
|------|-------------|
| `gpu_search(query, case_sensitive?)` | Exact-text pattern search. Use when `case_sensitive` matters. |
| `gpu_semantic_search(query, top_k?)` | Meaning-based search. Returns scored chunks with file + line range. |
| `gpu_index(directory)` | Rebuild pattern index (e.g. after large refactor). |
| `gpu_semantic_index(directory)` | Rebuild semantic embedding cache. Run once on first use, or when files change significantly. |
| `gpu_update_file(filepath)` | Re-index one file after editing (pattern index only). |
| `gpu_stats()` | Show VRAM usage for both indexes. |

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
Measured as direct Python calls (no MCP transport overhead).

| Query | Matches | gpu-search (RTX 4060) | ripgrep warm | ripgrep cold |
|-------|---------|----------------------|--------------|--------------|
| `ICodeEditor` | 428 files | **47ms** | ~110ms | ~6,300ms |
| `createTextModel` | 95 files | **10ms** | ~135ms | ~790ms |
| `disposeOnReturn` | 3 files | **8ms** | ~200ms | ~200ms |

- **gpu-search is 2–4× faster than ripgrep on warm searches** — no disk I/O ever after initial index load.
- ripgrep reads from OS file cache every time; gpu-search reads from VRAM (272 GB/s bandwidth).
- Cold start (first run): ripgrep reads from disk (~6s); gpu-search indexes once at startup (~3s), then every search is sub-100ms.
- Unique to gpu-search: **semantic search** (meaning-based, no exact match needed) and **blast radius** (which files import your result).

## Architecture

```
gpu_service/
├── gpu_index.py            # GpuFileIndex — VRAM byte loading, vectorized pattern search, watcher
├── gpu_semantic_index.py   # SemanticIndex — bge-small chunking, embedding, disk cache, cosine search
└── mcp_server.py           # FastMCP server — search_code router, all tools, watchdog, CLI
```

## License

MIT
