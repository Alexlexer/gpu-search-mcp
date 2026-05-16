# Limitations

This document describes known limitations of gpu-search-mcp. Understanding these helps you use
the tool safely and set appropriate expectations for API clients such as LegacyLens.

---

## Dependency impact

The dependency graph is built with **regex/heuristic import parsing**, not a full compiler or
language server. It handles common patterns across Python, JS/TS, Go, Rust, Java, C#, and
Ruby, but has inherent limitations.

**What it does:**
- Parses `import`, `require`, `using`, and similar statements with per-language regex rules.
- For C#: maps `using` namespaces, type declarations, and base/interface names to files.
- Builds a sparse in-VRAM graph and computes transitive reachability via BFS.

**What it does not do:**
- It is **not compiler-accurate**. Dynamic imports (`importlib`, `require()` with variables),
  conditional imports, and code generation are not resolved.
- **C# does not use Roslyn.** Heuristics cover common patterns but miss aliased types, global
  usings (C# 10+), and generated partial classes.
- False positives are possible (graph thinks A imports B when it doesn't).
- False negatives are possible (graph misses an import that exists).

**HTTP confidence field:**
The `/dependency/impact` endpoint returns a `confidence` field:
- `"medium"` — dependency graph is built and the analysis ran.
- `"low"` — dependency graph is not built; call `dep_index` first.

**Impact reasons:**
Each item in `/dependency/impact` `impactedFiles[]` may include an optional `reason` field,
such as `"imports namespace MyApp.Services"`, `"references type UserService"`, or
`"implements interface IUserService"`. These reasons explain why the heuristic graph linked
the files. They are useful advisory context for review UIs, but they are **not proof** of a
compiler-accurate dependency.

**How clients should treat this:** use `dep_impact` results as **advisory context**, not as
proof of impact. Treat the impacted file list as "files worth reviewing", not as a guaranteed
blast radius.

---

## Semantic search

Semantic search uses a pre-trained embedding model (BAAI/bge-small-en-v1.5, ~90 MB) to find
code by meaning rather than by exact token match.

**Known constraints:**
- The model must be downloaded and cached on first use (~90 MB, requires internet access).
  Set `HF_HUB_OFFLINE=1` if you have the model cached locally and want to prevent network access.
- The semantic index must be built with `gpu_semantic_index` before semantic queries work.
  The `/stats` endpoint `capabilities.semanticSearch` is `true` only when the index is ready.
- Semantic results are **approximate** — embedding similarity, not exact matching. The model
  may retrieve conceptually related but contextually irrelevant results.
- Natural-language query results should be verified by reading the returned files directly.
- The index covers files in 40-line chunks with 8-line overlap. Long functions or files may
  produce chunk boundaries that split context.

---

## GPU / CPU behavior

The server selects the best available compute backend automatically:

| Backend | Condition | Notes |
|---|---|---|
| CUDA (NVIDIA) | `torch.cuda.is_available()` | Full acceleration, tested on RTX 4060 |
| MPS (Apple Silicon) | `torch.backends.mps.is_available()` | Metal GPU, tested on M-series |
| CPU | Fallback | Always works; slower for large repos |

**CPU fallback caveats:**
- Pattern search is still fast on CPU (reads cached data, no disk I/O per query).
- Semantic embedding on first build is noticeably slower — minutes instead of seconds for
  large repos.
- VRAM usage stats will show 0 MB when running on CPU.

**Performance depends on:**
- Repo size (file count and total bytes).
- GPU VRAM capacity (pattern index uses 2× corpus bytes).
- Whether a warm cache exists (subsequent starts skip rebuild).

---

## Secret handling

**What is protected by default:**
- `.env` files are excluded from indexing unless `--allow-env-files` is passed.
- Search output is redacted — common credential patterns (API keys, bearer tokens, passwords,
  connection strings, PEM keys, AWS access key IDs) are replaced with `[REDACTED]`.

**What is not guaranteed:**
- Redaction is **best-effort pattern matching**, not a DLP scanner. Novel credential formats,
  obfuscated strings, or keys embedded in complex data structures may not be caught.
- When `--allow-env-files` is active, raw secret bytes live in VRAM for the lifetime of the
  server process even though search output is still redacted.
- The redaction layer applies to search snippets returned to the caller. It does not modify
  files on disk.

**Recommended practice:** do not index repositories containing production secrets. Treat
redaction as a safety net for accidental exposure, not as a compliance control.

---

## HTTP mode

HTTP mode exposes all search and read tools as a local JSON API.

**Binding:**
- Default bind is `127.0.0.1` (localhost only).
- `0.0.0.0` requires `--host 0.0.0.0` and logs a warning.

**Access control:**
- There is **no authentication** on HTTP endpoints.
- **Do not expose this API to the public internet.**
- For access from another machine on the same network, use [Tailscale](https://tailscale.com/)
  or configure local firewall rules so only trusted hosts can reach the port.

**Path safety:**
- All file-reading endpoints validate paths against configured/indexed roots.
- Path traversal (`../`) and absolute paths outside roots are rejected with HTTP 400.
- This prevents reading arbitrary local files via the API.

---

## Large repositories

**Initial indexing:**
- Pattern index build time scales with total corpus size (~3–10 s for a 285 MB repo on an RTX
  4060, longer on CPU).
- Semantic index build time scales with file count and chunk count (minutes for large repos
  without GPU).

**Caching:**
- Pattern and dependency indexes are cached under `.gpu-search-cache/` and validated on
  restart. A warm cache loads in < 1 s.
- Semantic cache is validated by model ID, chunk parameters, and directory fingerprint.
  A stale cache is deleted and rebuilt on the next `gpu_semantic_index` call.

**Recommendations:**
- Run `gpu-search-bench` on your own repo to measure actual latency.
- Use `top_k` to limit semantic results for expensive queries.
- Use `context_mode="compact"` to reduce token usage on large result sets.
- Avoid calling `dep_impact` on heavily-imported core utilities — they may list hundreds of
  dependent files.
