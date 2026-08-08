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

Exact pattern search is now **out-of-core**. Indexed source bytes are packed under
`.gpusearch/`, read through a replaceable storage backend, and verified through a bounded
reusable buffer pool. The default configuration uses two 2 MiB buffers, so exact-search
working VRAM is no longer proportional to repository size.

**CPU fallback caveats:**
- Pattern search remains fully functional on CPU, but normal queries read candidate ranges
  from the packed corpus instead of relying on a permanently resident full-corpus tensor.
- The current default candidate selector scans every chunk, so physical reads can approach
  the full packed corpus for each exact query until candidate pruning is implemented.
- Semantic embedding on first build is noticeably slower — minutes instead of seconds for
  large repos.
- VRAM usage stats will show 0 MB when running on CPU.

**Performance depends on:**
- Repo size, file count, and total packed corpus bytes.
- Chunk size and reusable buffer count.
- Storage backend and filesystem/NVMe performance.
- Candidate selectivity. The current `AllChunksCandidateSelector` returns every chunk, so
  large-corpus exact-search latency still scales with corpus size.
- Whether a warm validated cache exists.

The initial out-of-core CUDA baseline on a synthetic 64 MiB corpus used 4 MiB of reusable
buffer VRAM instead of the legacy implementation's ~128 MiB corpus allocation, but dense
all-chunk queries were 28–46% slower. Treat out-of-core search as a memory-scalability
foundation; candidate pruning and pipeline overlap are still performance work.

---

## Secret handling

**What is protected by default:**
- `.env` files are excluded from indexing unless `--allow-env-files` is passed.
- Search output is redacted — common credential patterns (API keys, bearer tokens, passwords,
  connection strings, PEM keys, AWS access key IDs) are replaced with `[REDACTED]`.

**What is not guaranteed:**
- Redaction is **best-effort pattern matching**, not a DLP scanner. Novel credential formats,
  obfuscated strings, or keys embedded in complex data structures may not be caught.
- When `--allow-env-files` is active, raw `.env` bytes are stored in the local packed corpus
  under `.gpusearch/` and may pass through host staging/device buffers while matching queries.
  Search output is still redacted, but the packed corpus itself contains the original bytes.
- The redaction layer applies to search snippets returned to the caller. It does not modify
  files on disk or sanitize the packed corpus.

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

**Exact-search architecture:**
- Source files are streamed into `.gpusearch/corpus.bin` during build/update and are not kept
  as a full raw + lowercase device corpus.
- `files.idx` stores stable file addressing and line metadata; `chunks.idx` stores stable
  chunk IDs and ranges.
- Normal exact queries read packed ranges through `FileStorageBackend` by default; `mmap` and
  explicit in-memory backends implement the same contract.
- The verifier is storage-agnostic, which leaves a clean future integration point for optional
  KvikIO/cuFile/direct-storage backends.

**Current scaling limitation:**
- The default candidate selector returns **100% of chunks**. Out-of-core search removes the
  VRAM-size ceiling, but it does not yet remove O(corpus-size) verification work per exact
  query. A selective trigram/ngram or equivalent candidate index is the main next scaling
  improvement.
- The current read → host-to-device copy → verification loop is synchronous. The buffer pool
  supports reusable allocations, but asynchronous double-buffered prefetch is not implemented.

**Initial indexing:**
- Pattern corpus build time scales with total source bytes and filesystem throughput.
- Semantic index build time scales with file count and chunk count (minutes for large repos
  without GPU).

**Caching:**
- Packed exact-search artifacts live under `.gpusearch/`; dependency and semantic cache
  artifacts and cache metadata live under `.gpu-search-cache/`.
- Pattern, dependency, and semantic identities are validated against SHA-256 source-content
  and producer/configuration metadata on restart.
- Multi-file cache updates use repository locking, temporary files, fsync, atomic promotion,
  rollback backups, stale-lock recovery, and interrupted-transaction detection where
  applicable. Packed pattern artifacts use versioned indexes and atomic replacement.
- Fingerprinting reads indexed source content, so validation cost scales with repository size;
  it avoids trusting mtime/size metadata across branch switches and worktrees.
- Watcher-storm and branch/worktree reconciliation coverage is still being expanded.

**Recommendations:**
- Run `gpu-search-bench` on your own repo to measure actual latency, candidate percentage,
  physical-read ratio, transfer cost, and bounded VRAM usage.
- Use `top_k` to limit semantic results for expensive queries.
- Use `context_mode="compact"` to reduce token usage on large result sets.
- Avoid calling `dep_impact` on heavily-imported core utilities — they may list hundreds of
  dependent files.
- For the current out-of-core exact path, expect large repositories to benefit most after a
  selective candidate index is added.
