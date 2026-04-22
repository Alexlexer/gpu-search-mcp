# GPU Search v0.1.0 — Plan

**Scope strategy:** safest cut — 4 items for v0.1.0, rest deferred to v0.2.0.
Rationale: land reliability first so later features ship on a stable base, then pick the highest-leverage LLM-ergonomics wins.

---

## v0.1.0 (ship first)

### 1. Reliability fixes (foundation)

Pre-req for everything else. Low risk, high stability value.

- **Bounded semantic update executor** — replace unbounded `on_modified` thread spawning with a singleton `ThreadPoolExecutor(max_workers=4)`.
- **Cache corruption recovery** — wrap `try_load_cache` so a bad `.npz` is logged + deleted, not a startup crash.
- **Graceful OOM** — `max_chunks` cap on semantic index (default 500k); surface as error in `gpu_stats` instead of VRAM exhaustion.
- **Verify lazy model load** — confirm `semantic._get_model()` is only called on first query, not in startup thread.

Files: `gpu_service/gpu_semantic_index.py`, `gpu_service/mcp_server.py`.

---

### 2. AST expansion + skeleton mode (shared tree-sitter investment)

Biggest token-quality win per hour of work. Both features share the same parser infrastructure — do them together.

**AST expansion:**
- GPU search returns match at byte offset.
- Tree-sitter parses the file, walks up from the offset to the enclosing function/class/interface.
- Output: syntactically complete block instead of `±10 lines` window that cuts brackets.

**Skeleton mode:**
- For large files (>N lines), strip function bodies, keep signatures + the matched block expanded.
- LLM sees the file's architecture for ~200 tokens instead of 15k.

**Design choices to settle:**
- Languages for v0.1.0: Python + TypeScript/JS (covers most use cases). Add more in v0.2.0.
- Fallback: if no parser for file type, use current line-window expansion.
- Config: `expand="ast"|"lines"|"skeleton"|"auto"` on search tools; `auto` = AST if parser exists + block <N lines, else skeleton for large files, else lines.

Dependency: `tree-sitter`, `tree-sitter-python`, `tree-sitter-typescript`.

Files: new `gpu_service/ast_expand.py`, `gpu_service/mcp_server.py` (wire into `search_code` output).

Risk: parser install complexity on Windows. Mitigation: graceful fallback if import fails.

---

### 3. Recency / git-status weighting

Cheapest win. Matches real developer intent — 90% of queries relate to recently touched code.

- On startup + periodically, read `git diff --name-only HEAD` and file mtimes.
- Scoring boost:
  - File in `git diff` (uncommitted): +0.3
  - File in last N commits: +0.15
  - File mtime within last hour: +0.1
- Boosts stack (capped at +0.4).
- Gate with `boost_recent=True` flag (default on).

Files: `gpu_service/gpu_index.py`, new `gpu_service/git_state.py`.

Risk: tiny — pure scoring tweak.

---

### 4. Hybrid search (headline feature)

Benefits from AST expansion for output quality.

- `search_code(query, mode="auto"|"pattern"|"semantic"|"hybrid")`, default `auto` keeps current routing.
- `hybrid`: run pattern + semantic in parallel threads.
- Normalize: pattern → `match_count / max_matches`, semantic → cosine (already 0–1).
- Merge by file path; files appearing in both get +0.2 boost.
- Output section: `Pattern matches` + `Semantically related` (deduped).
- Only run semantic half when `semantic._model` is warm (avoid cold-start latency regression).

Files: `gpu_service/mcp_server.py`, possibly `gpu_service/gpu_index.py` for parallel entry point.

---

## v0.2.0 (deferred)

Not in v0.1.0 — worth doing, but expands scope beyond "safest cut."

| # | Feature | Why deferred |
|---|---|---|
| 1 | Regex support (`gpu_regex_search`) | Useful but niche; no blocker for core UX. |
| 2 | Cache management (`gpu_cache_list/clear/verify`) | Quality-of-life for power users; not load-bearing. |
| 3 | Performance profiling (`gpu_benchmark`, p50/p95/p99 in stats) | Nice for proving perf claims; not user-facing. |
| 4 | Multi-pass tool split (`gpu_find_symbols` / `gpu_read_block`) | Redesigns tool surface. Additive not breaking, but bigger scope. |
| 5 | VRAM session pagination | Introduces statefulness (session GC, TTL). More impactful once AST output is richer. |

---

## Suggested implementation order

1. Reliability fixes → merge, verify stability
2. AST expansion + skeleton mode → merge
3. Recency weighting → merge
4. Hybrid search → merge, cut v0.1.0
