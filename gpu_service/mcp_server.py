"""
GPU-accelerated codebase search MCP server.
Files are pre-loaded into RTX VRAM; searches run as parallel CUDA kernels.

Usage: python mcp_server.py [--directory PATH]
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

VERSION = "0.1.0"
CONFIG_PATH = Path.home() / ".gpu-search-config.json"


class _SafeStderr:
    """Best-effort stderr wrapper for MCP hosts that close/redirect stderr oddly.

    Some Windows MCP clients can leave the child process with a stderr handle
    that raises OSError(22, "Invalid argument") from background threads. Logging
    must never abort indexing work, so ignore stderr write/flush failures.
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, data):
        try:
            return self._wrapped.write(data)
        except OSError:
            return 0

    def flush(self):
        try:
            return self._wrapped.flush()
        except OSError:
            return None

    def isatty(self):
        try:
            return self._wrapped.isatty()
        except OSError:
            return False

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


sys.stderr = _SafeStderr(sys.stderr)


def _load_config_dirs() -> list[str]:
    """Read directories from ~/.gpu-search-config.json."""
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return [d for d in data.get("directories", []) if os.path.isdir(d)]
    except Exception:
        pass
    return []


def _save_config_dirs(dirs: list[str]):
    """Persist directory list to ~/.gpu-search-config.json."""
    try:
        existing: list[str] = []
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            existing = data.get("directories", [])
        merged = list(dict.fromkeys(existing + dirs))  # deduplicate, preserve order
        CONFIG_PATH.write_text(json.dumps({"directories": merged}, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[gpu-search] Could not save config: {e}", file=sys.stderr, flush=True)

sys.path.insert(0, os.path.dirname(__file__))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from mcp.server.fastmcp import FastMCP, Context
from ast_expand import read_block, skeleton_file
from git_state import GitState
from redact import redact

INDEXED_EXTS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.c', '.cpp', '.h',
    '.hpp', '.java', '.cs', '.rb', '.php', '.swift', '.kt', '.json', '.yaml',
    '.yml', '.toml', '.md', '.txt', '.html', '.css', '.scss', '.sql', '.sh',
    '.bat', '.ps1', '.cfg', '.ini', '.xml',
    # .env excluded by default — pass --allow-env-files to opt in
}

# Set to True via --allow-env-files CLI flag or allow_env_files key in config JSON
_ALLOW_ENV_FILES: bool = False

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build',
    '.next', '.nuxt', 'target', 'bin', 'obj', '.idea', '.vscode', '.mypy_cache'
}

_DEP_EXTS = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".cs", ".rb"}
MAX_CHUNKS = 500_000

# Static limitation strings shared between /dependency/impact and /stats.
_DEP_LIMITATIONS = [
    "Dependency impact is based on import/type/name heuristics and is not compiler-accurate.",
    "C# analysis does not use Roslyn — namespace, type, and base/interface name heuristics are used instead.",
]

_GLOBAL_LIMITATIONS = [
    "Dependency impact is heuristic, not compiler-accurate.",
    "Secret redaction is best-effort pattern matching, not a DLP scanner.",
    "CPU fallback is slower than CUDA/MPS for large repositories.",
    "Semantic search requires model download/cache on first use.",
    "HTTP mode is local-first — do not expose to the public internet.",
]


class _LazyService:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None
        self._lock = threading.Lock()

    def _get(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)


def _make_index():
    from gpu_index import GpuFileIndex
    return GpuFileIndex()


def _get_effective_indexed_exts() -> set:
    return INDEXED_EXTS | ({'.env'} if _ALLOW_ENV_FILES else set())


def _make_semantic():
    from gpu_semantic_index import SemanticIndex
    return SemanticIndex()


def _make_deps():
    from gpu_dep_index import DepIndex
    return DepIndex()


mcp = FastMCP("gpu-search")
index = _LazyService(_make_index)
semantic = _LazyService(_make_semantic)
deps = _LazyService(_make_deps)
git_state = GitState()

# Background indexing status — updated by worker threads
_bg_status: dict[str, str] = {"pattern": "", "deps": "", "semantic": ""}

# Roots we've already loaded semantic cache for (avoid re-loading on every call)
_loaded_roots: set[str] = set()
_http_roots: list[str] = []

# Bounded executor for watchdog-triggered semantic updates — prevents thread storms on rapid saves
_semantic_update_executor = ThreadPoolExecutor(max_workers=4)


class _Debouncer:
    """Coalesces rapid file-change events into a single delayed call per key.

    Rapid saves (e.g., editor auto-save every second) would otherwise trigger
    repeated full-corpus rebuilds. With a 2s window, the rebuild fires once
    after the user stops saving.
    """

    def __init__(self, delay: float = 2.0):
        self._delay = delay
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def submit(self, key: str, fn, *args):
        """Schedule fn(*args) after delay, cancelling any previous pending call for key."""
        with self._lock:
            existing = self._pending.get(key)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self._delay, self._fire, args=(key, fn, args))
            self._pending[key] = timer
            timer.start()

    def _fire(self, key: str, fn, args):
        with self._lock:
            self._pending.pop(key, None)
        fn(*args)


_debouncer = _Debouncer(delay=2.0)


def _auto_load_semantic(ctx):
    """Load semantic cache for any client workspace root not yet loaded."""
    try:
        import asyncio
        session = ctx.request_context.session
        result = asyncio.get_event_loop().run_until_complete(session.list_roots())
        for root in result.roots:
            path = root.uri.replace("file:///", "").replace("file://", "")
            # Normalize Windows drive letter
            if len(path) >= 2 and path[1] == "/" :
                path = path[0].upper() + ":" + path[1:]
            path = os.path.abspath(path)
            if path in _loaded_roots or not os.path.isdir(path):
                continue
            _loaded_roots.add(path)
            def _load(p=path):
                already_has_index = semantic.stats()["chunks"] > 0
                s = semantic.try_load_cache(p) if not already_has_index else None
                if s is None and already_has_index:
                    try:
                        s = semantic.index_directory(p, append=True)
                    except Exception:
                        pass
                if s:
                    _bg_status["semantic"] = f"done: {s['chunks']} chunks ({s['vram_mb']} MB)"
                    print(f"[gpu-search] Auto-loaded semantic cache for {p}: {s['chunks']} chunks", file=sys.stderr, flush=True)
            threading.Thread(target=_load, daemon=True).start()
    except Exception:
        pass


def _make_observer():
    if sys.platform == "darwin":
        return PollingObserver()
    return Observer()


def _is_skipped_path(fpath: str) -> bool:
    return any(part in SKIP_DIRS for part in Path(fpath).parts)


class _Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory or _is_skipped_path(event.src_path):
            return
        ext = Path(event.src_path).suffix.lower()
        effective = _get_effective_indexed_exts()
        if ext in effective:
            _debouncer.submit(
                f"pattern:{event.src_path}",
                index.update_file, event.src_path, _ALLOW_ENV_FILES,
            )
            _debouncer.submit(
                f"semantic:{event.src_path}",
                lambda p=event.src_path: _semantic_update_executor.submit(semantic.update_file, p),
            )
        if ext in _DEP_EXTS:
            _debouncer.submit(f"deps:{event.src_path}", deps.update_file, event.src_path)

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if event.is_directory or _is_skipped_path(event.src_path):
            return
        _debouncer.submit(
            f"pattern:{event.src_path}",
            index.update_file, event.src_path, _ALLOW_ENV_FILES,
        )
        _debouncer.submit(
            f"semantic:{event.src_path}",
            lambda p=event.src_path: _semantic_update_executor.submit(semantic.update_file, p),
        )


def _is_natural_language(query: str) -> bool:
    """Heuristic: multi-word prose queries go to semantic; identifiers/symbols go to pattern."""
    words = query.strip().split()
    if len(words) <= 1:
        return False
    # If every word is purely alphabetic (no underscores, dots, parens) it reads as prose
    return all(w.isalpha() for w in words)


_MAX_FILES = 8
_MAX_MATCHES_PER_FILE = 2
_SNIPPET_CHARS = 300
_MAX_EXPAND_LINES = 60   # blocks larger than this get truncated in search output


def _expand_block(filepath: str, line: int) -> tuple[str, int, int] | None:
    """Return (code, start_line, end_line) for the AST block at line, or None on error."""
    try:
        code, start, end = read_block(filepath, line)
        n = end - start + 1
        if n > _MAX_EXPAND_LINES:
            code = "\n".join(code.splitlines()[:_MAX_EXPAND_LINES]) + \
                   f"\n... ({n - _MAX_EXPAND_LINES} more lines — call gpu_read_block for full)"
            end = start + _MAX_EXPAND_LINES - 1
        return code, start, end
    except Exception:
        return None


def _context_mode_opts(context_mode: str) -> tuple[bool, int, int]:
    mode = (context_mode or "normal").lower()
    if mode == "compact":
        return False, 5, 1
    if mode == "full":
        return True, _MAX_FILES, _MAX_MATCHES_PER_FILE
    return True, _MAX_FILES, _MAX_MATCHES_PER_FILE


def _format_pattern_results(results: list, stats: dict, expand: bool = True,
                            context_mode: str = "normal") -> str:
    if not results:
        return None
    base = stats['base_dir'] or ""
    total_files = results[0].get('_total_files', len(results)) if results else 0
    lines = [f"Pattern: {total_files} files matched:"]
    expand, max_files, max_matches = _context_mode_opts(context_mode)
    for r in results[:max_files]:
        rel = os.path.relpath(r['file'], base) if base else r['file']
        reason = r.get("reason", "exact token match" + (" + recent git activity" if git_state.boost(r["file"]) else ""))
        lines.append(f"\n{rel}:  reason: {reason}")
        if expand:
            seen: set[tuple[int, int]] = set()
            for m in r['matches'][:max_matches]:
                block = _expand_block(r['file'], m['line'])
                if block:
                    code, start, end = block
                    if (start, end) in seen:
                        continue
                    seen.add((start, end))
                    lines.append(f"  L{start}–{end}:")
                    lines.extend(f"    {ln}" for ln in redact(code).splitlines())
                else:
                    lines.append(f"  {m['line']}: {redact(m['content'])}")
        else:
            shown = r['matches'][:max_matches]
            more = len(r['matches']) - len(shown)
            for m in shown:
                lines.append(f"  L{m['line']}: {redact(m['content'])[:160]}")
            if more:
                lines.append(f"  ... {more} more matches")
    if total_files > max_files:
        lines.append(f"\n... {total_files - max_files} more files not shown — refine your query")
    return "\n".join(lines)


def _format_semantic_results(results: list, query: str, s: dict, expand: bool = True,
                             context_mode: str = "normal") -> str:
    if not results:
        return None
    base = s["base_dir"] or ""
    lines = [f"Semantic: {len(results)} matches for '{query}':"]
    expand, _, _ = _context_mode_opts(context_mode)
    for r in results:
        rel = os.path.relpath(r["file"], base) if base else r["file"]
        reason = r.get("reason", "semantic match" + (" + recent git activity" if git_state.boost(r["file"]) else ""))
        if expand:
            block = _expand_block(r["file"], r["start_line"])
            if block:
                code, start, end = block
                lines.append(f"\n[{r['score']}] {rel} L{start}–{end}: reason: {reason}")
                lines.extend(f"  {ln}" for ln in redact(code).splitlines())
                continue
        snippet_len = 160 if context_mode == "compact" else 300
        lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}: reason: {reason}")
        lines.append(redact(r["snippet"][:snippet_len]))
    return "\n".join(lines)


def _format_hybrid_results(
    pattern_results: list, semantic_results: list, query: str,
    p_stats: dict, s_stats: dict, context_mode: str = "normal",
) -> str:
    base = p_stats.get("base_dir") or s_stats.get("base_dir") or ""

    # Apply git boosts
    for r in pattern_results:
        r["_boost"] = git_state.boost(r["file"])
        r["reason"] = "exact token match" + (" + recent git activity" if r["_boost"] else "")
    for r in semantic_results:
        boost = git_state.boost(r["file"])
        r["score"] = round(r["score"] + boost, 4)
        r["reason"] = "semantic match" + (" + recent git activity" if boost else "")

    pattern_results.sort(key=lambda r: r["_boost"], reverse=True)
    semantic_results.sort(key=lambda r: r["score"], reverse=True)

    # Semantic-only: files not already surfaced by pattern search
    pattern_files = {r["file"] for r in pattern_results}
    semantic_only = [r for r in semantic_results if r["file"] not in pattern_files]

    lines = []

    if pattern_results:
        total = pattern_results[0].get("_total_files", len(pattern_results))
        lines.append(f"Pattern ({total} files matched):")
        _, max_files, max_matches = _context_mode_opts(context_mode)
        for r in pattern_results[:max_files]:
            rel = os.path.relpath(r["file"], base) if base else r["file"]
            shown = r["matches"][:max_matches]
            more = len(r["matches"]) - len(shown)
            lines.append(f"\n{rel}: reason: {r.get('reason')}")
            for m in shown:
                lines.append(f"  {m['line']}: {redact(m['content'])}")
            if more:
                lines.append(f"  ... {more} more matches")
        if total > max_files:
            lines.append(f"\n... {total - max_files} more files — refine your query")

    if semantic_only:
        if lines:
            lines.append("")
        lines.append(f"Semantically related ({len(semantic_only)} files not in pattern results):")
        max_sem = 5 if context_mode != "compact" else 3
        for r in semantic_only[:max_sem]:
            rel = os.path.relpath(r["file"], base) if base else r["file"]
            lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}: reason: {r.get('reason')}")
            lines.append(redact(r["snippet"][:(160 if context_mode == "compact" else _SNIPPET_CHARS)]))

    if not lines:
        return None
    return "\n".join(lines)


def _do_pattern_search(query: str) -> list:
    return index.search(query)


def _do_semantic_search(query: str, top_k: int) -> list:
    return semantic.search(query, top_k=top_k)


def _append_blast_radius(out: str, pattern_results: list) -> str:
    if deps.stats()["files"] == 0 or not pattern_results:
        return out
    matched_files = list({r["file"] for r in pattern_results[:5]})
    all_impact: dict[str, int] = {}
    for f in matched_files:
        for item in deps.impact(f):
            if item["file"] not in all_impact or item["hops"] < all_impact[item["file"]]:
                all_impact[item["file"]] = item["hops"]
    if not all_impact:
        return out
    base = index.stats()["base_dir"]
    out += f"\n\nBlast radius ({len(all_impact)} files import these results):"
    direct = [f for f, h in sorted(all_impact.items(), key=lambda x: x[1]) if h == 1]
    indirect = [f for f, h in sorted(all_impact.items(), key=lambda x: x[1]) if h > 1]
    if direct:
        out += "\n  Direct: " + ", ".join(os.path.relpath(f, base) for f in direct[:5])
    if indirect:
        out += f"\n  Indirect: {len(indirect)} more files"
    return out


def _relpath_for_api(fpath: str, base: str) -> str:
    return os.path.relpath(fpath, base) if base else fpath


def _pattern_structured(results: list, stats: dict, context_mode: str = "compact") -> list[dict]:
    base = stats.get("base_dir") or ""
    _, max_files, max_matches = _context_mode_opts(context_mode)
    out: list[dict] = []
    for r in results[:max_files]:
        boost = git_state.boost(r["file"])
        reason = r.get("reason") or "exact token match" + (" + recent git activity" if boost else "")
        for m in r.get("matches", [])[:max_matches]:
            out.append({
                "file": _relpath_for_api(r["file"], base),
                "absoluteFile": r["file"],
                "lineStart": m["line"],
                "lineEnd": m["line"],
                "score": round(1.0 + boost, 4),
                "reason": reason,
                "snippet": redact(m.get("content", ""))[:300],
                "engine": "pattern",
            })
    return out


def _semantic_structured(results: list, stats: dict, context_mode: str = "compact") -> list[dict]:
    base = stats.get("base_dir") or ""
    limit = 160 if context_mode == "compact" else 600
    out: list[dict] = []
    for r in results:
        boost = git_state.boost(r["file"])
        reason = r.get("reason") or "semantic match" + (" + recent git activity" if boost else "")
        out.append({
            "file": _relpath_for_api(r["file"], base),
            "absoluteFile": r["file"],
            "lineStart": r.get("start_line"),
            "lineEnd": r.get("end_line"),
            "score": r.get("score"),
            "reason": reason,
            "snippet": redact(r.get("snippet", "")[:limit]),
            "engine": "semantic",
        })
    return out


def _http_search_structured(query: str, top_k: int = 5, mode: str = "auto",
                            context_mode: str = "compact") -> dict:
    semantic_candidate = mode in {"semantic", "hybrid"} or (mode == "auto" and _is_natural_language(query))
    pattern_ready = index.stats()["files"] > 0
    semantic_stats = semantic.stats()
    semantic_ready = semantic_stats["chunks"] > 0
    effective = "semantic" if mode == "auto" and semantic_candidate and semantic_ready else mode
    if effective == "auto":
        effective = "pattern"

    pattern_results: list = []
    semantic_results: list = []
    if effective in {"pattern", "hybrid"} and pattern_ready:
        pattern_results = index.search(query)
        for r in pattern_results:
            r["reason"] = "exact token match" + (" + recent git activity" if git_state.boost(r["file"]) else "")
        pattern_results.sort(key=lambda r: git_state.boost(r["file"]), reverse=True)
    if effective in {"semantic", "hybrid"} and semantic_ready:
        semantic_results = semantic.search(query, top_k=top_k)
        for r in semantic_results:
            boost = git_state.boost(r["file"])
            r["score"] = round(r["score"] + boost, 4)
            r["reason"] = "semantic match" + (" + recent git activity" if boost else "")
        semantic_results.sort(key=lambda r: r["score"], reverse=True)

    base = index.stats().get("base_dir") or semantic_stats.get("base_dir") or ""
    pattern_files = {r["file"] for r in pattern_results}
    results = _pattern_structured(pattern_results, {"base_dir": base}, context_mode)
    results.extend(_semantic_structured(
        [r for r in semantic_results if effective != "hybrid" or r["file"] not in pattern_files],
        {"base_dir": base},
        context_mode,
    ))
    return {
        "query": query,
        "mode": effective,
        "contextMode": context_mode,
        "results": results,
    }


@mcp.tool()
def search_code(query: str, top_k: int = 5, mode: str = "auto",
                context_mode: str = "normal", ctx: Context = None) -> str:
    """Search code by exact identifier or natural language.
    mode: 'auto' (default) routes identifier→pattern, prose→semantic;
          'hybrid' runs both in parallel and merges results;
          'pattern' or 'semantic' forces a specific engine.
    context_mode: 'compact' returns file/line/snippet/reason; 'normal' expands
          small AST blocks; 'full' is the least compressed.
    Use this for ALL searches."""
    semantic_candidate = mode in {"semantic", "hybrid"} or (mode == "auto" and _is_natural_language(query))

    if ctx is not None and semantic_candidate:
        _auto_load_semantic(ctx)

    pattern_ready = False
    if mode in {"pattern", "hybrid"} or not semantic_candidate:
        pattern_ready = index.stats()["files"] > 0

    semantic_ready = False
    if semantic_candidate:
        semantic_stats = semantic.stats()
        semantic_ready = semantic_stats["chunks"] > 0

    # Resolve effective mode
    if mode == "auto":
        effective = "semantic" if _is_natural_language(query) and semantic_ready else "pattern"
    else:
        effective = mode

    # ── Hybrid ────────────────────────────────────────────────────────────────
    if effective == "hybrid":
        if not pattern_ready and not semantic_ready:
            return "No index found. Call gpu_index and gpu_semantic_index first."

        p_results: list = []
        s_results: list = []

        def _run_p():
            if pattern_ready:
                p_results.extend(_do_pattern_search(query))

        def _run_s():
            if semantic_ready:
                try:
                    s_results.extend(_do_semantic_search(query, top_k))
                except Exception:
                    pass

        pt = threading.Thread(target=_run_p)
        st = threading.Thread(target=_run_s)
        pt.start()
        st.start()
        pt.join()
        st.join()

        out = _format_hybrid_results(p_results, s_results, query, index.stats(), semantic.stats(), context_mode)
        if out and p_results:
            out = _append_blast_radius(out, p_results)
        return out or f"No results for '{query}'"

    # ── Semantic ──────────────────────────────────────────────────────────────
    if effective == "semantic":
        if not semantic_ready:
            return semantic.semantic_unavailable_message()
        try:
            results = semantic.search(query, top_k=top_k)
        except Exception:
            return semantic.semantic_unavailable_message()
        for r in results:
            r["score"] = round(r["score"] + git_state.boost(r["file"]), 4)
        results.sort(key=lambda r: r["score"], reverse=True)
        out = _format_semantic_results(results, query, semantic.stats(), context_mode=context_mode)
        return out or f"No semantic matches for '{query}'"

    # ── Pattern (default) ────────────────────────────────────────────────────
    if not pattern_ready:
        return "No index found. Call gpu_index (and optionally gpu_semantic_index) with your project directory first."
    results = _do_pattern_search(query)
    results.sort(key=lambda r: git_state.boost(r["file"]), reverse=True)
    for r in results:
        r["reason"] = "exact token match" + (" + recent git activity" if git_state.boost(r["file"]) else "")
    out = _format_pattern_results(results, index.stats(), context_mode=context_mode)
    if not out:
        return f"No matches for '{query}'"
    out = _append_blast_radius(out, results)
    return out


@mcp.tool()
def gpu_search(query: str, case_sensitive: bool = False) -> str:
    """Exact-text pattern search. Use only when case_sensitive control is needed; otherwise use search_code."""
    stats = index.stats()
    if stats['files'] == 0:
        return "No files indexed. Call gpu_index with your project directory first."

    results = index.search(query, case_sensitive=case_sensitive)
    out = _format_pattern_results(results, stats)
    return out or f"No matches for '{query}'"


@mcp.tool()
async def gpu_index(directory: str, append: bool = False) -> str:
    """Load a directory into GPU VRAM for pattern search. Runs in background; call gpu_stats to check. append=True for multi-root."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    def _do():
        _bg_status["pattern"] = f"indexing {directory}..."
        stats = index.index_directory(directory, append=append, allow_env_files=_ALLOW_ENV_FILES)
        _bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"

    threading.Thread(target=_do, daemon=True).start()
    return f"Pattern indexing started for {directory} — call gpu_stats to check progress."


@mcp.tool()
def gpu_stats() -> str:
    """Show index status and VRAM usage for all indexes."""
    p = index.stats()
    s = semantic.stats()
    d = deps.stats()
    g_modified = len(git_state._modified)
    g_recent   = len(git_state._recent)
    lines = [
        f"Pattern index:  {p['files']} files, {p['vram_mb']} MB, cache={p.get('cache', 'n/a')}  ({p['base_dir'] or 'none'})",
        f"Semantic index: {s['chunks']} chunks, {s['vram_mb']} MB  ({s['base_dir'] or 'not built'})",
        f"Dep graph:      {d['files']} files, {d['edges']} edges, cache={d.get('cache', 'n/a')}  ({d['base_dir'] or 'not built'})",
        f"Git state:      {g_modified} modified, {g_recent} recently-committed files",
    ]
    if _bg_status["pattern"]:
        lines.append(f"Pattern status:   {_bg_status['pattern']}")
    if _bg_status["deps"]:
        lines.append(f"Deps status:      {_bg_status['deps']}")
    if _bg_status["semantic"]:
        lines.append(f"Semantic status:  {_bg_status['semantic']}")
    if s.get("chunks_capped"):
        lines.append(f"Semantic WARNING: chunk cap ({MAX_CHUNKS:,}) hit — index is partial")
    if s.get("embed_progress"):
        lines.append(f"Embed progress:   {s['embed_progress']}")
    if s.get("last_error"):
        lines.append(f"Semantic ERROR:   {s['last_error']}")
    elif s.get("model_error"):
        lines.append(f"Semantic ERROR:   {s['model_error']}")
    return "\n".join(lines)


@mcp.tool()
def gpu_update_file(filepath: str) -> str:
    """Re-index a specific file after editing it (keeps VRAM in sync)."""
    index.update_file(filepath)
    return f"Updated: {filepath}"


@mcp.tool()
def gpu_read_block(filepath: str, line: int) -> str:
    """Read the AST-expanded block (function/class) that contains the given line number. Pass a line from search results to get the full syntactically-complete context instead of a raw snippet."""
    if not os.path.isfile(filepath):
        return f"File not found: {filepath}"
    code, start, end = read_block(filepath, line)
    base = index.stats().get("base_dir") or os.path.dirname(filepath)
    rel = os.path.relpath(filepath, base)
    return f"{rel} L{start}–{end}:\n```\n{code}```"


@mcp.tool()
def gpu_skeleton(filepath: str, match_lines: list[int] = None) -> str:
    """Return a code skeleton of a file with unexpanded function bodies folded to '...'. Pass match_lines (from search results) to keep those blocks fully expanded. Useful for understanding a large file's structure without reading all N thousand lines."""
    if not os.path.isfile(filepath):
        return f"File not found: {filepath}"
    result = skeleton_file(filepath, match_lines)
    if result is None:
        # Unsupported file type — return a plain line count summary instead
        try:
            lines = open(filepath, encoding="utf-8", errors="replace").readlines()
            return f"No AST parser for this file type ({len(lines)} lines). Use Read tool to view it directly."
        except Exception as e:
            return f"Could not read {filepath}: {e}"
    base = index.stats().get("base_dir") or os.path.dirname(filepath)
    rel = os.path.relpath(filepath, base)
    return f"Skeleton of {rel}:\n```\n{result}```"


@mcp.tool()
async def gpu_semantic_index(directory: str, append: bool = False, force: bool = False) -> str:
    """Build semantic embedding cache for a directory (bge-small-en-v1.5). Runs in background; cache persists across restarts. append=True for multi-root, force=True to rebuild."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    if not append:
        semantic.reset(base_dir=os.path.abspath(directory))

    def _do():
        try:
            _bg_status["semantic"] = f"embedding {directory}..."
            stats = semantic.index_directory(directory, append=append, force=force)
            _bg_status["semantic"] = f"done: {stats['chunks']} chunks ({stats['vram_mb']} MB)"
            print(f"[gpu-search] Semantic index ready: {stats['chunks']} chunks ({stats['vram_mb']} MB VRAM)", file=sys.stderr, flush=True)
        except Exception as e:
            _bg_status["semantic"] = f"ERROR: {e}"
            print(f"[gpu-search] Semantic index FAILED: {e}", file=sys.stderr, flush=True)
            if not semantic.stats().get("model_error"):
                import traceback
                traceback.print_exc(file=sys.stderr)

    threading.Thread(target=_do, daemon=True).start()
    return f"Semantic indexing started for {directory} — call gpu_stats to check progress."


@mcp.tool()
async def gpu_add_directory(directory: str) -> str:
    """Add a directory to the permanent startup config so it auto-indexes on every future launch. Also indexes immediately."""
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    _save_config_dirs([directory])

    def _do_pattern():
        git_state.add_root(directory)
        append = index.stats()["files"] > 0
        _bg_status["pattern"] = f"indexing {directory}..."
        stats = index.index_directory(directory, append=append, allow_env_files=_ALLOW_ENV_FILES)
        _bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"

    def _do_deps():
        try:
            append = deps.stats()["files"] > 0
            _bg_status["deps"] = f"indexing {directory}..."
            dep_stats = deps.index_directory(directory, append=append)
            _bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"
        except Exception as e:
            _bg_status["deps"] = f"ERROR: {e}"
            print(f"[gpu-search] Dep index FAILED: {e}", file=sys.stderr, flush=True)

    def _do_semantic():
        s = semantic.try_load_cache(directory)
        if s is None:
            _bg_status["semantic"] = f"no semantic cache for {directory} — run gpu_semantic_index to build it"
        else:
            _bg_status["semantic"] = f"done: {s['chunks']} chunks ({s['vram_mb']} MB)"
            _loaded_roots.add(directory)

    threading.Thread(target=_do_pattern, daemon=True).start()
    threading.Thread(target=_do_deps, daemon=True).start()
    threading.Thread(target=_do_semantic, daemon=True).start()
    return f"Added '{directory}' to startup config. Indexing started — call gpu_stats to check progress."


@mcp.tool()
def dep_impact(filepath: str) -> str:
    """CALL BEFORE EDITING. Returns every file that transitively imports the given file, grouped by hop distance. Call dep_index first."""
    s = deps.stats()
    if s["files"] == 0:
        if _bg_status.get("deps", "").startswith("indexing "):
            return f"Dependency graph is still building: {_bg_status['deps']}"
        return "Dependency graph not built. Call dep_index with your project directory first."

    results = deps.impact(filepath)
    if not results:
        rel = os.path.relpath(filepath, s["base_dir"]) if s["base_dir"] else filepath
        return f"Nothing in the project imports '{rel}' — safe to change."

    base = s["base_dir"]
    by_hop: dict[int, list[str]] = {}
    for r in results:
        by_hop.setdefault(r["hops"], []).append(r)

    _MAX_PER_HOP = 20
    lines = [f"Impact of changing '{os.path.relpath(filepath, base)}' ({len(results)} affected files):"]
    for hop in sorted(by_hop):
        files = by_hop[hop]
        label = "Direct importers" if hop == 1 else f"Indirect (depth {hop})"
        shown = files[:_MAX_PER_HOP]
        lines.append(f"\n{label} ({len(files)} files):")
        for item in shown:
            reason = item.get("reason")
            suffix = f" — {reason}" if reason else ""
            lines.append(f"  {os.path.relpath(item['file'], base)}{suffix}")
        if len(files) > _MAX_PER_HOP:
            lines.append(f"  ... {len(files) - _MAX_PER_HOP} more")
    return "\n".join(lines)


@mcp.tool()
def dep_imports(filepath: str) -> str:
    """Show all project files directly imported by the given file."""
    s = deps.stats()
    if s["files"] == 0:
        if _bg_status.get("deps", "").startswith("indexing "):
            return f"Dependency graph is still building: {_bg_status['deps']}"
        return "Dependency graph not built. Call dep_index with your project directory first."

    imports = deps.direct_imports(filepath)
    base = s["base_dir"]
    rel = os.path.relpath(filepath, base) if base else filepath
    if not imports:
        return f"'{rel}' has no tracked project imports."
    lines = [f"'{rel}' directly imports:"]
    for f in imports:
        lines.append(f"  {os.path.relpath(f, base) if base else f}")
    return "\n".join(lines)


@mcp.tool()
async def dep_index(directory: str, append: bool = False) -> str:
    """Build import dependency graph (Python/JS/TS/Go/Rust/Java/C#/Ruby). Runs in background. Required before dep_impact/dep_imports. append=True for multi-root."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    def _do():
        try:
            _bg_status["deps"] = f"indexing {directory}..."
            s = deps.index_directory(directory, append=append)
            _bg_status["deps"] = f"done: {s['files']} files, {s['edges']} edges"
        except Exception as e:
            _bg_status["deps"] = f"ERROR: {e}"
            print(f"[gpu-search] Dep index FAILED: {e}", file=sys.stderr, flush=True)

    threading.Thread(target=_do, daemon=True).start()
    return f"Dep graph indexing started for {directory} — call gpu_stats to check progress."


@mcp.tool()
def gpu_semantic_search(query: str, top_k: int = 5) -> str:
    """Semantic search by meaning (GPU cosine similarity). Use search_code for most queries; use this when you need explicit top_k control."""
    s = semantic.stats()
    if s["chunks"] == 0:
        if s.get("model_error"):
            return s["model_error"]
        return "No semantic index found. Call gpu_semantic_index with your project directory first."
    try:
        results = semantic.search(query, top_k=top_k)
    except Exception:
        return semantic.semantic_unavailable_message()
    if not results:
        return f"No results for '{query}'"

    base = s["base_dir"]
    lines = [f"Semantic: {len(results)} matches for '{query}':"]
    for r in results:
        rel = os.path.relpath(r["file"], base) if base else r["file"]
        lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}")
        lines.append(redact(r["snippet"][:_SNIPPET_CHARS]))
    return "\n".join(lines)


@mcp.prompt()
def search_codebase(query: str) -> str:
    """Search the indexed codebase for any identifier, symbol, or concept."""
    return (
        f"Use search_code('{query}') to find relevant code. "
        "If the results are exact matches, read the surrounding context with the Read tool. "
        "If no matches are found, try a broader or natural-language rephrasing."
    )


@mcp.prompt()
def before_edit(filepath: str) -> str:
    """Understand the blast radius of a file before changing it."""
    return (
        f"Before editing '{filepath}', call dep_impact('{filepath}') to see every file "
        "that transitively imports it. Review the direct importers (hop 1) carefully — "
        "those are the files most likely to break. Then make your edit and check those files for regressions."
    )


@mcp.prompt()
def explore_feature(description: str) -> str:
    """Find where a feature or concept is implemented across the codebase."""
    return (
        f"To locate '{description}' in the codebase:\n"
        f"1. Call search_code('{description}') — this will route to semantic search if it reads as natural language.\n"
        "2. For the top results, use dep_imports(filepath) to understand what each file depends on.\n"
        "3. Use dep_impact(filepath) on key files to see what else references them.\n"
        "This gives you both the implementation site and its full call graph."
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="gpu-search-mcp: GPU-accelerated code search MCP server")
    parser.add_argument("--directory", "-d", action="append", dest="directories",
                        metavar="DIR", help="Directory to index (repeat for multi-root)")
    parser.add_argument(
        "--allow-env-files", action="store_true", default=False,
        help="Opt in to indexing .env files (excluded by default for security)",
    )
    parser.add_argument("--http", action="store_true", help="Run HTTP API instead of MCP stdio")
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind host. Defaults to 127.0.0.1; use 0.0.0.0 only explicitly.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    return parser.parse_args(argv)


def _prepare_startup(args):
    global _ALLOW_ENV_FILES
    if args.allow_env_files:
        _ALLOW_ENV_FILES = True
        INDEXED_EXTS.add('.env')
        print("[gpu-search] WARNING: .env indexing enabled — search results may contain secrets",
              file=sys.stderr, flush=True)

    cli_dirs = [os.path.abspath(d) for d in (args.directories or [])]
    config_dirs = _load_config_dirs()
    if not cli_dirs:
        cli_dirs = [os.getcwd()]
    extra_dirs = [d for d in config_dirs if d not in cli_dirs and os.path.isdir(d)]

    cli_targets = [t for t in cli_dirs if os.path.isdir(t)]
    all_targets = cli_targets + extra_dirs

    if cli_dirs:
        _save_config_dirs(cli_dirs)
    print(f"[gpu-search] Index targets: {cli_targets}  |  Semantic cache: {all_targets}",
          file=sys.stderr, flush=True)
    return cli_targets, all_targets


def _start_indexes(cli_targets: list[str], all_targets: list[str]):

    # Import and construct services before spawning startup worker threads.
    # The services share heavy dependencies (torch/numpy); importing them from
    # multiple background threads can leave early MCP tool calls waiting on the
    # import lock for a long time. Eager construction makes stdio invocations
    # responsive as soon as the server accepts requests.
    index._get()
    deps._get()
    semantic._get()

    observer = _make_observer()
    for target in cli_targets:
        observer.schedule(_Watcher(), target, recursive=True)

    def _startup_pattern():
        for i, target in enumerate(cli_targets):
            try:
                _bg_status["pattern"] = f"indexing {target}..."
                stats = index.index_directory(target, append=(i > 0), allow_env_files=_ALLOW_ENV_FILES)
                _bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"
                print(f"[gpu-search] Pattern index: {stats['indexed']} files ({stats['vram_mb']} MB) from {target}",
                      file=sys.stderr)
            except Exception as e:
                _bg_status["pattern"] = f"ERROR: {e}"
                print(f"[gpu-search] Pattern index FAILED: {e}", file=sys.stderr, flush=True)

    def _startup_deps():
        for i, target in enumerate(cli_targets):
            try:
                _bg_status["deps"] = f"indexing {target}..."
                dep_stats = deps.index_directory(target, append=(i > 0))
                _bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"
                print(f"[gpu-search] Dep graph: {dep_stats['files']} files, {dep_stats['edges']} edges from {target}",
                      file=sys.stderr)
            except Exception as e:
                _bg_status["deps"] = f"ERROR: {e}"
                print(f"[gpu-search] Dep index FAILED: {e}", file=sys.stderr, flush=True)

    def _startup_semantic():
        for i, target in enumerate(all_targets):
            s = semantic.try_load_cache(target) if i == 0 else semantic.merge_cache(target)
            if s:
                _loaded_roots.add(os.path.abspath(target))
                print(f"[gpu-search] Semantic cache merged: {s['chunks']} chunks ({s['vram_mb']} MB) total",
                      file=sys.stderr, flush=True)

    for target in cli_targets:
        git_state.add_root(target)

    threading.Thread(target=_startup_pattern, daemon=True).start()
    threading.Thread(target=_startup_deps, daemon=True).start()
    threading.Thread(target=_startup_semantic, daemon=True).start()
    observer.daemon = True
    observer.start()
    return observer


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _active_roots() -> list[str]:
    roots = list(_http_roots)
    for stats in (index.stats(), semantic.stats(), deps.stats()):
        base = stats.get("base_dir")
        if base:
            roots.append(base)
    # Deduplicate after resolving existing roots.
    out: list[str] = []
    for root in roots:
        try:
            resolved = str(Path(root).resolve())
        except Exception:
            continue
        if resolved not in out:
            out.append(resolved)
    return out


def _require_under_root(filepath: str) -> str:
    if not filepath:
        raise ValueError("Missing filepath")
    resolved = Path(filepath).resolve()
    roots = [Path(r).resolve() for r in _active_roots()]
    if not roots:
        raise ValueError("No indexed roots configured")
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError("Path outside indexed roots")
    return str(resolved)


def _infer_language(filepath: str) -> str:
    return {
        ".cs": "csharp", ".py": "python", ".ts": "typescript",
        ".tsx": "typescriptreact", ".js": "javascript", ".jsx": "javascriptreact",
        ".json": "json", ".sql": "sql",
    }.get(Path(filepath).suffix.lower(), "text")


def _csharp_ast_available() -> bool:
    try:
        import tree_sitter_c_sharp  # noqa: F401
        return True
    except ImportError:
        return False


class _HttpApi(BaseHTTPRequestHandler):
    server_version = "gpu-search-mcp/0.1"

    def log_message(self, fmt, *args):
        print(f"[gpu-search-http] {self.address_string()} - {fmt % args}", file=sys.stderr)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0"))
        if n <= 0:
            return {}
        return json.loads(self.rfile.read(n).decode("utf-8"))

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            return _json_response(self, 200, {"ok": True, "version": VERSION})
        if path == "/stats":
            p_stats = index.stats()
            s_stats = semantic.stats()
            d_stats = deps.stats()
            return _json_response(self, 200, {
                "pattern": p_stats,
                "semantic": s_stats,
                "dependency": d_stats,
                "status": _bg_status,
                "capabilities": {
                    "patternSearch": p_stats["files"] > 0,
                    "semanticSearch": s_stats["chunks"] > 0,
                    "dependencyImpact": d_stats["files"] > 0,
                    "csharpAst": _csharp_ast_available(),
                    "httpStructuredResponses": True,
                },
                "limitations": _GLOBAL_LIMITATIONS,
            })
        return _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        try:
            payload = self._read_json()
            path = urlparse(self.path).path
            if path == "/search/code":
                mode = payload.get("mode", "auto")
                context_mode = payload.get("contextMode", payload.get("context_mode", "normal"))
                result = search_code(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode=mode,
                    context_mode=context_mode,
                )
                structured = _http_search_structured(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode=mode,
                    context_mode=context_mode,
                )
                return _json_response(self, 200, {"result": result, **structured})
            if path == "/search/hybrid":
                context_mode = payload.get("contextMode", payload.get("context_mode", "normal"))
                result = search_code(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="hybrid",
                    context_mode=context_mode,
                )
                structured = _http_search_structured(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="hybrid",
                    context_mode=context_mode,
                )
                return _json_response(self, 200, {"result": result, **structured})
            if path == "/search/semantic":
                context_mode = payload.get("contextMode", payload.get("context_mode", "normal"))
                result = search_code(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="semantic",
                    context_mode=context_mode,
                )
                structured = _http_search_structured(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="semantic",
                    context_mode=context_mode,
                )
                return _json_response(self, 200, {"result": result, **structured})
            if path == "/read/block":
                safe_path = _require_under_root(payload.get("filepath", payload.get("file", "")))
                line = int(payload.get("line", 1))
                if not os.path.isfile(safe_path):
                    return _json_response(self, 400, {"error": f"File not found: {safe_path}"})
                code, line_start, line_end = read_block(safe_path, line)
                base = index.stats().get("base_dir") or os.path.dirname(safe_path)
                rel = os.path.relpath(safe_path, base)
                result_str = f"{rel} L{line_start}–{line_end}:\n```\n{code}```"
                return _json_response(self, 200, {
                    "result": result_str,
                    "file": rel,
                    "absoluteFile": safe_path,
                    "lineStart": line_start,
                    "lineEnd": line_end,
                    "content": redact(code),
                    "language": _infer_language(safe_path),
                })
            if path == "/read/skeleton":
                safe_path = _require_under_root(payload.get("filepath", payload.get("file", "")))
                match_lines = payload.get("matchLines")
                if not os.path.isfile(safe_path):
                    return _json_response(self, 400, {"error": f"File not found: {safe_path}"})
                base = index.stats().get("base_dir") or os.path.dirname(safe_path)
                rel = os.path.relpath(safe_path, base)
                skel = skeleton_file(safe_path, match_lines)
                if skel is None:
                    try:
                        with open(safe_path, encoding="utf-8", errors="replace") as fh:
                            n_lines = len(fh.readlines())
                        result_str = (
                            f"No AST parser for this file type ({n_lines} lines)."
                            " Use Read tool to view it directly."
                        )
                    except Exception as exc:
                        result_str = f"Could not read {safe_path}: {exc}"
                    content_out = None
                else:
                    result_str = f"Skeleton of {rel}:\n```\n{skel}```"
                    content_out = redact(skel)
                return _json_response(self, 200, {
                    "result": result_str,
                    "file": rel,
                    "absoluteFile": safe_path,
                    "content": content_out,
                    "matchLines": match_lines or [],
                    "language": _infer_language(safe_path),
                })
            if path == "/dependency/impact":
                safe_path = _require_under_root(payload.get("filepath", payload.get("file", "")))
                result = dep_impact(safe_path)
                dep_stats = deps.stats()
                base = dep_stats.get("base_dir") or ""
                rel = os.path.relpath(safe_path, base) if base else safe_path
                warnings: list[str] = []
                if dep_stats["files"] > 0:
                    confidence = "medium"
                    try:
                        impact_list = deps.impact(safe_path)
                        impacted_files = [
                            {
                                "file": os.path.relpath(r["file"], base) if base else r["file"],
                                "absoluteFile": r["file"],
                                "hops": r["hops"],
                                **({"reason": r["reason"]} if r.get("reason") else {}),
                            }
                            for r in impact_list
                        ]
                    except Exception:
                        impacted_files = []
                    if not impacted_files:
                        warnings.append(
                            "No files in the dependency graph import this path."
                        )
                else:
                    confidence = "low"
                    impacted_files = []
                    warnings.append(
                        "Dependency graph not built. Call dep_index first to build it."
                    )
                return _json_response(self, 200, {
                    "result": result,
                    "file": rel,
                    "absoluteFile": safe_path,
                    "confidence": confidence,
                    "analysisMode": "heuristic",
                    "limitations": _DEP_LIMITATIONS,
                    "warnings": warnings,
                    "impactedFiles": impacted_files,
                })
            return _json_response(self, 404, {"error": "not found"})
        except ValueError as e:
            return _json_response(self, 400, {"error": str(e)})
        except Exception as e:
            return _json_response(self, 500, {"error": str(e)})


def _start_http(args, cli_targets: list[str], all_targets: list[str]):
    global _http_roots
    if args.host == "0.0.0.0":
        print("[gpu-search] WARNING: HTTP binding to 0.0.0.0; prefer 127.0.0.1 or Tailscale-only firewall rules.",
              file=sys.stderr, flush=True)
    _http_roots = [str(Path(t).resolve()) for t in cli_targets]
    _start_indexes(cli_targets, all_targets)
    httpd = ThreadingHTTPServer((args.host, args.port), _HttpApi)
    print(f"[gpu-search] HTTP API listening on http://{args.host}:{args.port}", file=sys.stderr, flush=True)
    httpd.serve_forever()


def _start_server(args):
    cli_targets, all_targets = _prepare_startup(args)
    if args.http:
        _start_http(args, cli_targets, all_targets)
        return
    _start_indexes(cli_targets, all_targets)

    mcp.run(transport="stdio")


def cli_main():
    """Entry point for the `gpu-search-mcp` CLI command."""
    _start_server(_parse_args())


if __name__ == "__main__":
    _start_server(_parse_args())
