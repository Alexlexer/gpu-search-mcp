# gpu-search-mcp

A GPU-accelerated codebase search server built as an [MCP](https://modelcontextprotocol.io/) tool. It loads your source files directly into RTX VRAM and runs searches as vectorized CUDA operations via PyTorch — no custom kernels, no native extensions.

## How it works

On startup the server builds two indexes in parallel:

1. **Pattern index** — every source file is read into VRAM as `uint8` tensors. Queries use a first-char GPU filter then a vectorized window check. Sub-millisecond for exact identifiers.
2. **Semantic index** — files are chunked into ~40-line windows and embedded with [`nomic-embed-code`](https://huggingface.co/nomic-ai/nomic-embed-code) in batches on GPU. The normalized `(N, 768)` float32 matrix lives in VRAM; queries are a single matmul.

A `watchdog` watcher keeps the pattern index in sync as you edit. The semantic index builds in a background thread so the server is immediately usable.

## Requirements

- Windows or Linux with an NVIDIA GPU (tested on RTX 4060)
- Python 3.10+
- CUDA toolkit compatible with your PyTorch build
- ~8 GB VRAM recommended for large monorepos

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mcp watchdog numpy sentence-transformers
```

## Usage

### Run the MCP server

```bash
# Index the current directory on startup
python gpu_service/mcp_server.py

# Or specify a directory
python gpu_service/mcp_server.py --directory C:/path/to/your/project
```

Both indexes build automatically. The semantic index prints a ready message when embedding finishes (~1–2 min depending on project size; model downloads ~500 MB on first run).

### MCP tools

**Use `search_code` for everything** — it auto-routes to the right backend:

| Query type | Example | Routes to |
|---|---|---|
| Identifier / symbol / literal | `"handleError"`, `"AUTH_TOKEN"` | Pattern search |
| Natural language | `"where is error handling middleware"` | Semantic search |

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
| `gpu_semantic_index(directory)` | Rebuild semantic index. |
| `gpu_update_file(filepath)` | Re-index one file after editing (pattern index only). |
| `gpu_stats()` | Show VRAM usage for both indexes. |

### Wire it into Claude Code

Add to your `claude_desktop_config.json` or Claude Code MCP settings:

```json
{
  "mcpServers": {
    "gpu-search": {
      "command": "python",
      "args": ["C:/dev/claudeUsesRtx/gpu_service/mcp_server.py", "--directory", "C:/your/project"]
    }
  }
}
```

Claude will automatically call `search_code` — no manual tool selection needed.

## File types indexed

`.py .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .java .cs .rb .php .swift .kt .json .yaml .yml .toml .md .txt .html .css .scss .sql .sh .bat .ps1 .cfg .ini .xml .env`

Directories skipped: `.git node_modules __pycache__ .venv venv dist build .next .nuxt target bin obj .idea .vscode .mypy_cache`

## Architecture

```
gpu_service/
├── gpu_index.py            # GpuFileIndex — VRAM byte loading, vectorized pattern search, watcher
├── gpu_semantic_index.py   # SemanticIndex — nomic-embed-code chunking, embedding, cosine search
└── mcp_server.py           # FastMCP server — search_code router, all tools, watchdog, CLI
```

## License

MIT
