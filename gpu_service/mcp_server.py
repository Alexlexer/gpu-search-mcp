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
from pathlib import Path

VERSION = "0.0.1"
CONFIG_PATH = Path.home() / ".gpu-search-config.json"


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
from gpu_index import GpuFileIndex, INDEXED_EXTS, SKIP_DIRS
from gpu_semantic_index import SemanticIndex, MAX_CHUNKS
from gpu_dep_index import DepIndex, _DEP_EXTS
from ast_expand import read_block, skeleton_file
from git_state import GitState

mcp = FastMCP("gpu-search")
index = GpuFileIndex()
semantic = SemanticIndex()
deps = DepIndex()
git_state = GitState()

# Background indexing status — updated by worker threads
_bg_status: dict[str, str] = {"pattern": "", "deps": "", "semantic": ""}

# Roots we've already loaded semantic cache for (avoid re-loading on every call)
_loaded_roots: set[str] = set()

# Bounded executor for watchdog-triggered semantic updates — prevents thread storms on rapid saves
_semantic_update_executor = ThreadPoolExecutor(max_workers=4)


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
        if not event.is_directory and not _is_skipped_path(event.src_path):
            ext = Path(event.src_path).suffix.lower()
            if ext in INDEXED_EXTS:
                index.update_file(event.src_path)
                _semantic_update_executor.submit(semantic.update_file, event.src_path)
            if ext in _DEP_EXTS:
                deps.update_file(event.src_path)

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if not event.is_directory and not _is_skipped_path(event.src_path):
            index.update_file(event.src_path)
            _semantic_update_executor.submit(semantic.update_file, event.src_path)


def _is_natural_language(query: str) -> bool:
    """Heuristic: multi-word prose queries go to semantic; identifiers/symbols go to pattern."""
    words = query.strip().split()
    if len(words) <= 1:
        return False
    # If every word is purely alphabetic (no underscores, dots, parens) it reads as prose
    return all(w.isalpha() for w in words)


_MAX_FILES = 8
_MAX_MATCHES_PER_FILE = 2
_SNIPPET_CHARS = 200


def _format_pattern_results(results: list, stats: dict) -> str:
    if not results:
        return None
    total_files = results[0].get('_total_files', len(results)) if results else 0
    lines = [f"Pattern: {total_files} files matched:"]
    for r in results[:_MAX_FILES]:
        rel = os.path.relpath(r['file'], stats['base_dir']) if stats['base_dir'] else r['file']
        shown = r['matches'][:_MAX_MATCHES_PER_FILE]
        more = len(r['matches']) - len(shown)
        lines.append(f"\n{rel}:")
        for m in shown:
            lines.append(f"  {m['line']}: {m['content']}")
        if more:
            lines.append(f"  ... {more} more matches")
    if total_files > _MAX_FILES:
        lines.append(f"\n... {total_files - _MAX_FILES} more files not shown — refine your query")
    return "\n".join(lines)


def _format_semantic_results(results: list, query: str, s: dict) -> str:
    if not results:
        return None
    base = s["base_dir"]
    lines = [f"Semantic: {len(results)} matches for '{query}':"]
    for r in results:
        rel = os.path.relpath(r["file"], base) if base else r["file"]
        lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}")
        lines.append(r["snippet"][:_SNIPPET_CHARS])
    return "\n".join(lines)


def _format_hybrid_results(
    pattern_results: list, semantic_results: list, query: str,
    p_stats: dict, s_stats: dict,
) -> str:
    base = p_stats.get("base_dir") or s_stats.get("base_dir") or ""

    # Apply git boosts
    for r in pattern_results:
        r["_boost"] = git_state.boost(r["file"])
    for r in semantic_results:
        r["score"] = round(r["score"] + git_state.boost(r["file"]), 4)

    pattern_results.sort(key=lambda r: r["_boost"], reverse=True)
    semantic_results.sort(key=lambda r: r["score"], reverse=True)

    # Semantic-only: files not already surfaced by pattern search
    pattern_files = {r["file"] for r in pattern_results}
    semantic_only = [r for r in semantic_results if r["file"] not in pattern_files]

    lines = []

    if pattern_results:
        total = pattern_results[0].get("_total_files", len(pattern_results))
        lines.append(f"Pattern ({total} files matched):")
        for r in pattern_results[:_MAX_FILES]:
            rel = os.path.relpath(r["file"], base) if base else r["file"]
            shown = r["matches"][:_MAX_MATCHES_PER_FILE]
            more = len(r["matches"]) - len(shown)
            lines.append(f"\n{rel}:")
            for m in shown:
                lines.append(f"  {m['line']}: {m['content']}")
            if more:
                lines.append(f"  ... {more} more matches")
        if total > _MAX_FILES:
            lines.append(f"\n... {total - _MAX_FILES} more files — refine your query")

    if semantic_only:
        if lines:
            lines.append("")
        lines.append(f"Semantically related ({len(semantic_only)} files not in pattern results):")
        for r in semantic_only[:5]:
            rel = os.path.relpath(r["file"], base) if base else r["file"]
            lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}")
            lines.append(r["snippet"][:_SNIPPET_CHARS])

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


@mcp.tool()
def search_code(query: str, top_k: int = 5, mode: str = "auto", ctx: Context = None) -> str:
    """Search code by exact identifier or natural language.
    mode: 'auto' (default) routes identifier→pattern, prose→semantic;
          'hybrid' runs both in parallel and merges results;
          'pattern' or 'semantic' forces a specific engine.
    Use this for ALL searches."""
    if ctx is not None:
        _auto_load_semantic(ctx)

    pattern_ready  = index.stats()["files"] > 0
    semantic_ready = semantic.stats()["chunks"] > 0 and semantic._model is not None

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
                s_results.extend(_do_semantic_search(query, top_k))

        pt = threading.Thread(target=_run_p)
        st = threading.Thread(target=_run_s)
        pt.start(); st.start()
        pt.join();  st.join()

        out = _format_hybrid_results(p_results, s_results, query, index.stats(), semantic.stats())
        if out and p_results:
            out = _append_blast_radius(out, p_results)
        return out or f"No results for '{query}'"

    # ── Semantic ──────────────────────────────────────────────────────────────
    if effective == "semantic":
        if not semantic_ready:
            return "Semantic index not ready. Call gpu_semantic_index first (or wait for model to load)."
        results = semantic.search(query, top_k=top_k)
        for r in results:
            r["score"] = round(r["score"] + git_state.boost(r["file"]), 4)
        results.sort(key=lambda r: r["score"], reverse=True)
        out = _format_semantic_results(results, query, semantic.stats())
        return out or f"No semantic matches for '{query}'"

    # ── Pattern (default) ────────────────────────────────────────────────────
    if not pattern_ready:
        return "No index found. Call gpu_index (and optionally gpu_semantic_index) with your project directory first."
    results = _do_pattern_search(query)
    results.sort(key=lambda r: git_state.boost(r["file"]), reverse=True)
    out = _format_pattern_results(results, index.stats())
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
        stats = index.index_directory(directory, append=append)
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
        f"Pattern index:  {p['files']} files, {p['vram_mb']} MB  ({p['base_dir'] or 'none'})",
        f"Semantic index: {s['chunks']} chunks, {s['vram_mb']} MB  ({s['base_dir'] or 'not built'})",
        f"Dep graph:      {d['files']} files, {d['edges']} edges  ({d['base_dir'] or 'not built'})",
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

    def _do():
        try:
            _bg_status["semantic"] = f"embedding {directory}..."
            stats = semantic.index_directory(directory, append=append, force=force)
            _bg_status["semantic"] = f"done: {stats['chunks']} chunks ({stats['vram_mb']} MB)"
            print(f"[gpu-search] Semantic index ready: {stats['chunks']} chunks ({stats['vram_mb']} MB VRAM)", file=sys.stderr, flush=True)
        except Exception as e:
            _bg_status["semantic"] = f"ERROR: {e}"
            print(f"[gpu-search] Semantic index FAILED: {e}", file=sys.stderr, flush=True)
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
        stats = index.index_directory(directory, append=append)
        _bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"

    def _do_deps():
        append = deps.stats()["files"] > 0
        _bg_status["deps"] = f"indexing {directory}..."
        dep_stats = deps.index_directory(directory, append=append)
        _bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"

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
        return "Dependency graph not built. Call dep_index with your project directory first."

    results = deps.impact(filepath)
    if not results:
        rel = os.path.relpath(filepath, s["base_dir"]) if s["base_dir"] else filepath
        return f"Nothing in the project imports '{rel}' — safe to change."

    base = s["base_dir"]
    by_hop: dict[int, list[str]] = {}
    for r in results:
        by_hop.setdefault(r["hops"], []).append(r["file"])

    _MAX_PER_HOP = 20
    lines = [f"Impact of changing '{os.path.relpath(filepath, base)}' ({len(results)} affected files):"]
    for hop in sorted(by_hop):
        files = by_hop[hop]
        label = "Direct importers" if hop == 1 else f"Indirect (depth {hop})"
        shown = files[:_MAX_PER_HOP]
        lines.append(f"\n{label} ({len(files)} files):")
        for f in shown:
            lines.append(f"  {os.path.relpath(f, base)}")
        if len(files) > _MAX_PER_HOP:
            lines.append(f"  ... {len(files) - _MAX_PER_HOP} more")
    return "\n".join(lines)


@mcp.tool()
def dep_imports(filepath: str) -> str:
    """Show all project files directly imported by the given file."""
    s = deps.stats()
    if s["files"] == 0:
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
        _bg_status["deps"] = f"indexing {directory}..."
        s = deps.index_directory(directory, append=append)
        _bg_status["deps"] = f"done: {s['files']} files, {s['edges']} edges"

    threading.Thread(target=_do, daemon=True).start()
    return f"Dep graph indexing started for {directory} — call gpu_stats to check progress."


@mcp.tool()
def gpu_semantic_search(query: str, top_k: int = 5) -> str:
    """Semantic search by meaning (GPU cosine similarity). Use search_code for most queries; use this when you need explicit top_k control."""
    s = semantic.stats()
    if s["chunks"] == 0:
        return "No semantic index found. Call gpu_semantic_index with your project directory first."
    if semantic._model is None:
        return "Semantic model still loading — try again in a few seconds."

    results = semantic.search(query, top_k=top_k)
    if not results:
        return f"No results for '{query}'"

    base = s["base_dir"]
    lines = [f"Semantic: {len(results)} matches for '{query}':"]
    for r in results:
        rel = os.path.relpath(r["file"], base) if base else r["file"]
        lines.append(f"\n[{r['score']}] {rel} L{r['start_line']}–{r['end_line']}")
        lines.append(r["snippet"][:_SNIPPET_CHARS])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", action="append", dest="directories",
                        metavar="DIR", help="Directory to index (repeat for multi-root)")
    args = parser.parse_args()

    cli_dirs = [os.path.abspath(d) for d in (args.directories or [])]
    config_dirs = _load_config_dirs()
    if not cli_dirs:
        cli_dirs = [os.getcwd()]
    # Deduplicate config dirs (exclude any already in cli_dirs)
    extra_dirs = [d for d in config_dirs if d not in cli_dirs and os.path.isdir(d)]

    # Pattern + dep indexing: CLI dirs only (fast for project roots, expensive for large repos)
    cli_targets = [t for t in cli_dirs if os.path.isdir(t)]
    # Semantic cache: load for all dirs (CLI + config) — instant .npz load, no embedding
    all_targets = cli_targets + extra_dirs

    if cli_dirs:
        _save_config_dirs(cli_dirs)
    print(f"[gpu-search] Index targets: {cli_targets}  |  Semantic cache: {all_targets}", file=sys.stderr, flush=True)

    observer = _make_observer()
    for target in cli_targets:
        observer.schedule(_Watcher(), target, recursive=True)

    def _startup_pattern():
        for i, target in enumerate(cli_targets):
            _bg_status["pattern"] = f"indexing {target}..."
            stats = index.index_directory(target, append=(i > 0))
            _bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"
            print(f"[gpu-search] Pattern index: {stats['indexed']} files ({stats['vram_mb']} MB) from {target}", file=sys.stderr)

    def _startup_deps():
        for i, target in enumerate(cli_targets):
            _bg_status["deps"] = f"indexing {target}..."
            dep_stats = deps.index_directory(target, append=(i > 0))
            _bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"
            print(f"[gpu-search] Dep graph: {dep_stats['files']} files, {dep_stats['edges']} edges from {target}", file=sys.stderr)

    def _startup_semantic():
        # Load model first so it's ready for queries while caches are still loading
        semantic._get_model()
        print(f"[gpu-search] Semantic model ready", file=sys.stderr, flush=True)
        # Merge all available caches (CLI dirs first, then config dirs) — .npz load, no embedding
        for i, target in enumerate(all_targets):
            s = semantic.try_load_cache(target) if i == 0 else semantic.merge_cache(target)
            if s:
                _loaded_roots.add(os.path.abspath(target))
                print(f"[gpu-search] Semantic cache merged: {s['chunks']} chunks ({s['vram_mb']} MB) total", file=sys.stderr, flush=True)

    for target in cli_targets:
        git_state.add_root(target)

    threading.Thread(target=_startup_pattern, daemon=True).start()
    threading.Thread(target=_startup_deps, daemon=True).start()
    threading.Thread(target=_startup_semantic, daemon=True).start()
    observer.daemon = True
    observer.start()

    mcp.run(transport="stdio")
