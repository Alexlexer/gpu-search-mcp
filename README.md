# gpu-search-mcp `v0.1.0`

A GPU-accelerated codebase search server built as an [MCP](https://modelcontextprotocol.io/) tool. It loads your source files directly into RTX VRAM and runs searches as vectorized CUDA operations via PyTorch — no custom kernels, no native extensions.

> **Status:** Working prototype, used daily on a single machine. Core search is solid; some features described below are best-effort (see [Limitations](#known-limitations)).
>
> **Release:** `v0.1.0` — [Release notes](docs/releases/v0.1.0.md) · [Changelog](CHANGELOG.md)

## Highlights

- GPU/CPU exact pattern search over a whole repo as one concatenated byte corpus.
- Semantic search with persistent embedding cache.
- Persistent pattern/dependency cache in `.gpu-search-cache/` for faster restarts.
- Dependency impact analysis for agent workflows (`dep_impact` before editing).
- C#/.NET-aware heuristics: `using`, namespaces, type declarations, base/interface names, AST/fallback block expansion.
- Low-token `compact` result mode with match reasons.
- MCP stdio mode for Claude/Codex and HTTP mode for local integrations such as LegacyLens.
- Structured HTTP DTOs for API clients, plus human-readable MCP-style strings.
- CI quality gates: pytest, Ruff, CPU/no-GPU compatibility, smoke test.

## How it works

On startup the server prepares three search-time data structures:

1. **Pattern index** — every indexed source file is read into VRAM as `uint8` tensors. Exact queries use a first-char GPU filter and vectorized window checks. Metadata and file bytes are persisted so the next launch can load a validated cache.
2. **Dependency graph** — project imports are parsed with regex/language heuristics into a sparse graph so the agent can answer "what imports this file?" before editing. This is **best-effort** and not compiler-accurate (see [dep graph limitations](#dependency-graph)).
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

Optional Tree-sitter grammars improve AST expansion/skeletons for Python, TypeScript/JavaScript, and C#:

```bash
pip install -e ".[ast]"
# or
uv sync --extra ast
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
search_code("authentication middleware", context_mode="compact")  # low-token file/line/reason output
```

`context_mode` can be `compact`, `normal` (default), or `full`. Compact mode returns file path, line range, a short snippet, and a ranking reason so agents can call `gpu_read_block` only for the files that matter.

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

Pattern and dependency indexes now persist under each project root:

```
.gpu-search-cache/
  pattern-index-v1.bin
  files-v1.json
  dep-graph-v1.json
  line-offsets-v1.bin
  cache-manifest.json
```

First run builds the indexes; later runs load the cache when file sizes/mtimes/hashes match. Changed files are refreshed and deleted files are removed from the next cache snapshot.

### HTTP mode

For non-MCP integrations (for example LegacyLens or a browser/client over Tailscale), run:

```bash
gpu-search-mcp --directory D:\repos\myapp --http
```

HTTP binds to `127.0.0.1` by default — local-first by design. It will not bind to `0.0.0.0` unless you explicitly pass `--host 0.0.0.0`. Use Tailscale or local network firewall rules if you need access from another machine. **Do not expose this API directly to the public internet.**

All HTTP file endpoints validate paths against configured/indexed roots. Requests for files outside those roots fail with `400` rather than reading arbitrary local paths.

#### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Version and liveness check |
| `/stats` | GET | Pattern, semantic, dependency, and background status |
| `/search/code` | POST | Auto-routed code search (pattern or semantic) |
| `/search/hybrid` | POST | Parallel pattern + semantic search, merged |
| `/search/semantic` | POST | Semantic-only (meaning-based) search |
| `/read/block` | POST | AST-expanded block at a given line |
| `/read/skeleton` | POST | Folded file outline with matched blocks expanded |
| `/dependency/impact` | POST | Files that transitively import the given file |
| `/scan/signals` | POST | Categorized repository signal scan (audit/onboarding) |

#### GET /health

```bash
curl http://127.0.0.1:8765/health
```

Response:

```json
{ "ok": true, "version": "0.1.0" }
```

#### GET /stats

```bash
curl http://127.0.0.1:8765/stats
```

Returns index sizes, VRAM usage, and background indexing progress.

#### POST /search/code

Request:

```json
{ "query": "UserService", "mode": "pattern", "contextMode": "compact", "topK": 5 }
```

- `mode`: `"auto"` (default), `"pattern"`, `"semantic"`, or `"hybrid"`
- `contextMode`: `"compact"` (short snippets), `"normal"` (AST-expanded), or `"full"`
- `topK`: maximum semantic results (default 5)

Response:

```json
{
  "result": "Pattern: 1 files matched:\n\nUserService.cs:  reason: exact token match\n  L1: public class UserService {}",
  "query": "UserService",
  "mode": "pattern",
  "contextMode": "compact",
  "results": [
    {
      "file": "src/UserService.cs",
      "absoluteFile": "D:\\repos\\app\\src\\UserService.cs",
      "lineStart": 42,
      "lineEnd": 42,
      "score": 1.0,
      "reason": "exact token match",
      "snippet": "public class UserService",
      "engine": "pattern"
    }
  ]
}
```

The `result` field is a human-readable string kept for backward compatibility. The `results` array is the stable structured integration surface for API clients such as LegacyLens.

#### POST /search/semantic

Request:

```json
{ "query": "where is authentication handled", "topK": 5, "contextMode": "compact" }
```

Response shape is identical to `/search/code` with `"mode": "semantic"` and `"engine": "semantic"` on each result.

#### POST /search/hybrid

Runs pattern and semantic search in parallel and merges results. Pattern hits appear first; semantic-only files follow. Files found by pattern search are not duplicated in the semantic section.

Request: same shape as `/search/code`.

#### POST /read/block

Request:

```json
{ "filepath": "D:\\repos\\app\\src\\UserService.cs", "line": 42 }
```

`filepath` may be an absolute path or a path relative to an indexed root.

Response:

```json
{
  "result": "src/UserService.cs L40–96:\n```\npublic class UserService { ... }```",
  "file": "src/UserService.cs",
  "absoluteFile": "D:\\repos\\app\\src\\UserService.cs",
  "lineStart": 40,
  "lineEnd": 96,
  "content": "public class UserService { ... }",
  "language": "csharp"
}
```

`language` is inferred from the file extension: `csharp`, `python`, `typescript`, `typescriptreact`, `javascript`, `javascriptreact`, `json`, `sql`, or `text`.

#### POST /read/skeleton

Request:

```json
{ "filepath": "D:\\repos\\app\\src\\UserService.cs", "matchLines": [42] }
```

Response:

```json
{
  "result": "Skeleton of src/UserService.cs:\n```\n...",
  "file": "src/UserService.cs",
  "absoluteFile": "D:\\repos\\app\\src\\UserService.cs",
  "content": "public class UserService {\n    ...  # 54 lines\n}",
  "matchLines": [42],
  "language": "csharp"
}
```

#### POST /dependency/impact

Request:

```json
{ "filepath": "D:\\repos\\app\\src\\UserService.cs" }
```

Response:

```json
{
  "result": "Impact of changing 'src/UserService.cs' (3 affected files): ...",
  "file": "src/UserService.cs",
  "absoluteFile": "D:\\repos\\app\\src\\UserService.cs",
  "impactedFiles": [
    {
      "file": "src/AuthController.cs",
      "absoluteFile": "D:\\repos\\app\\src\\AuthController.cs",
      "hops": 1,
      "reason": "imports namespace MyApp.Services"
    },
    {
      "file": "src/UserController.cs",
      "absoluteFile": "D:\\repos\\app\\src\\UserController.cs",
      "hops": 1,
      "reason": "references type UserService"
    }
  ]
}
```

`reason` is optional heuristic metadata. It explains why the graph linked a file (for example
`imports module settings`, `references type UserService`, or `implements interface IUserService`),
but it is not compiler-accurate proof of impact.

If the dependency graph has not been built, `impactedFiles` is `[]` and `result` explains how to build it.

#### 400 — path outside indexed roots

Any file endpoint that receives a path outside an indexed root returns:

```json
{ "error": "Path outside indexed roots" }
```

This applies to `../` traversal, absolute paths to other directories, and any path not contained within a configured root.

#### LegacyLens integration notes

The full API contract is documented in [`docs/openapi/gpu-search-mcp.openapi.yaml`](docs/openapi/gpu-search-mcp.openapi.yaml) (OpenAPI 3.1.0).

- The structured fields (`results`, `file`, `absoluteFile`, `lineStart`, `lineEnd`, `content`, `language`, `impactedFiles`) are the **stable integration surface**. LegacyLens should consume these fields directly.
- The `result` string in every response is the original MCP-style human-readable output. It is kept for backward compatibility and will not be removed, but clients must not parse it — its format is unspecified.
- Use `/scan/signals` for bulk audit-style scans — it runs many category searches in one call and returns categorized, bounded results. See [`docs/signal-scan.md`](docs/signal-scan.md).
- HTTP mode is local-first. Default bind is `127.0.0.1`. Use Tailscale or local network rules if accessing from another machine. Do not expose this API directly to the public internet.
- HTTP endpoints reject file reads outside indexed roots — path traversal returns 400.

Example curl commands:

```bash
curl http://127.0.0.1:8765/health

curl -X POST http://127.0.0.1:8765/search/code ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"UserService\",\"mode\":\"pattern\",\"contextMode\":\"compact\"}"

curl -X POST http://127.0.0.1:8765/read/block ^
  -H "Content-Type: application/json" ^
  -d "{\"filepath\":\"D:\\\\repos\\\\app\\\\src\\\\UserService.cs\",\"line\":42}"

curl -X POST http://127.0.0.1:8765/dependency/impact ^
  -H "Content-Type: application/json" ^
  -d "{\"filepath\":\"D:\\\\repos\\\\app\\\\src\\\\UserService.cs\"}"

curl -X POST http://127.0.0.1:8765/scan/signals ^
  -H "Content-Type: application/json" ^
  -d "{\"categories\":[\"legacy-dotnet\",\"sql\"],\"topKPerSignal\":5}"
```

## File types indexed

`.py .js .ts .tsx .jsx .go .rs .c .cpp .h .hpp .java .cs .rb .php .swift .kt .json .yaml .yml .toml .md .txt .html .css .scss .sql .sh .bat .ps1 .cfg .ini .xml`

`.env` is excluded by default. Pass `--allow-env-files` to opt in.

Directories skipped: `.git node_modules __pycache__ .venv venv dist build .next .nuxt target bin obj .idea .vscode .mypy_cache`

## Quality gates

CI runs on Python 3.10 and 3.12 with CPU-only PyTorch so contributors do not need CUDA in GitHub Actions:

- `ruff check gpu_service/ tests/`
- `pytest tests/`
- smoke test without semantic model download

GPU behavior is covered by runtime fallback: CUDA is used when available, Apple MPS on supported Macs, otherwise CPU.

## Benchmark

> **Example results** — collected by the author on a specific machine. Your results will depend on hardware, OS, and repo size.

Run your own JSON benchmark:

```bash
gpu-search-bench --directory D:\repos\vscode --queries benchmarks/queries.json --output results.json
```

The JSON includes machine info, repo size, file count, index build time, VRAM usage, p50/p95/p99 direct Python latency, and ripgrep warm-call timing when `rg` is installed.

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
├── bench.py                # JSON benchmark CLI
├── git_state.py            # Recency weighting from git diff/commit history
├── redact.py               # Secret redaction for search output
└── mcp_server.py           # FastMCP + HTTP API — routing, watchers, startup flow
```

## Known limitations

Full details in [docs/limitations.md](docs/limitations.md). Summary:

### Dependency graph

The dependency graph is built with **regex/heuristic import parsing**, not a full compiler. It handles common patterns in Python, JS/TS, Go, Rust, Java, C#, and Ruby, but will miss:
- Dynamic imports (`importlib`, `require()` with variables)
- Conditional imports
- Generated or bundled code

Use `dep_impact` results as a starting point, not a guarantee. The `/dependency/impact` HTTP
endpoint returns a `confidence` field (`"medium"` when the graph is ready, `"low"` when
not built), a `limitations` list, and optional impacted-file `reason` strings so API clients
can surface advisory context to users.

### Token usage

`search_code` prefers AST-expanded blocks over raw line windows, which improves context quality but can be expensive on large repos. A single call can cost hundreds to a few thousand tokens.

Workarounds:
- Use `top_k` to limit results: `search_code("query", top_k=3)`
- Use `gpu_search` for exact identifiers (pattern results are shorter)
- Use `mode="pattern"` when you do not want semantic expansion
- Avoid `dep_impact` on highly-imported files (core utilities can list hundreds of dependents)

### Pattern + dependency cache accuracy

Pattern and dependency caches are invalidated by file list, size, mtime, and content hash checks. If the manifest is stale or corrupt, the server falls back to rebuilding from disk.

### Tree-sitter coverage

AST expansion and skeleton mode target Python, TypeScript/JavaScript, and C# when the matching Tree-sitter grammar is installed. C# also has a brace-matching fallback for common class/method/controller-action blocks. Unsupported file types fall back to line-window snippets.

### C# dependency analysis

C# dependency analysis is still best-effort, but now understands `using` statements, namespaces, type declarations, and base/interface names. It can map common namespace imports and interface implementations to project files; compiler-accurate Roslyn integration is still future work.

### Secret redaction is best-effort

The redaction layer catches common patterns but is not a comprehensive DLP scanner. Do not rely on it as your only secret protection — keep secrets out of source files in the first place.

### Safe mode / audit mode

`.env` exclusion, secret redaction, localhost-only HTTP defaults, and HTTP root validation are implemented. A single explicit `--safe-mode` / `--audit-index` command is still future work.

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
