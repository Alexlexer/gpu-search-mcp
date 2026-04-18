# gpu-search-mcp

A GPU-accelerated codebase search server built as an [MCP](https://modelcontextprotocol.io/) tool. It loads your source files directly into RTX VRAM and runs searches as vectorized CUDA operations via PyTorch — no custom kernels, no native extensions.

## How it works

1. On startup, the server walks a target directory and reads every source file into GPU VRAM as `uint8` tensors.
2. Search queries are encoded to bytes, lowercased if needed, and matched against in-VRAM buffers using a two-pass approach: a first-character filter narrows candidates, then a vectorized window check confirms full matches — all on the GPU.
3. A `watchdog` file-system watcher keeps VRAM in sync as you edit files.

Because the entire indexed corpus lives in VRAM, repeated searches skip disk I/O entirely.

## Requirements

- Windows or Linux with an NVIDIA GPU (tested on RTX 4060)
- Python 3.10+
- CUDA toolkit compatible with your PyTorch build
- ~8 GB VRAM recommended for large monorepos

## Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install mcp watchdog numpy
```

## Usage

### Run the MCP server

```bash
# Index the current directory on startup
python gpu_service/mcp_server.py

# Or specify a directory
python gpu_service/mcp_server.py --directory C:/path/to/your/project
```

### MCP tools exposed

| Tool | Description |
|------|-------------|
| `gpu_index(directory)` | Load a project directory into VRAM. Re-call to refresh. |
| `gpu_search(query, case_sensitive?)` | Search the indexed codebase from VRAM. |
| `gpu_stats()` | Show files in VRAM and memory usage. |
| `gpu_update_file(filepath)` | Re-index a single file after editing. |

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

Then in Claude: call `gpu_index` once to load a project, then use `gpu_search` instead of grep.

## File types indexed

`.py .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .java .cs .rb .php .swift .kt .json .yaml .yml .toml .md .txt .html .css .scss .sql .sh .bat .ps1 .cfg .ini .xml .env`

Directories skipped: `.git node_modules __pycache__ .venv venv dist build .next .nuxt target bin obj .idea .vscode .mypy_cache`

## Architecture

```
gpu_service/
├── gpu_index.py    # GpuFileIndex class — VRAM loading, tensor search, file watcher updates
└── mcp_server.py   # FastMCP server — tool definitions, watchdog integration, CLI entrypoint
```

## License

MIT
