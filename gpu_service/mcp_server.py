"""
GPU-accelerated codebase search MCP server.
Files are pre-loaded into RTX VRAM; searches run as parallel CUDA kernels.

Usage: python mcp_server.py [--directory PATH]
"""
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mcp.server.fastmcp import FastMCP
from gpu_index import GpuFileIndex, INDEXED_EXTS
from gpu_semantic_index import SemanticIndex
from gpu_dep_index import DepIndex, _DEP_EXTS
from pathlib import Path

mcp = FastMCP("gpu-search")
index = GpuFileIndex()
semantic = SemanticIndex()
deps = DepIndex()

# Background indexing status — updated by worker threads
_bg_status: dict[str, str] = {"pattern": "", "deps": ""}


class _Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            ext = Path(event.src_path).suffix.lower()
            if ext in INDEXED_EXTS:
                index.update_file(event.src_path)
                threading.Thread(target=semantic.update_file, args=(event.src_path,), daemon=True).start()
            if ext in _DEP_EXTS:
                deps.update_file(event.src_path)

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if not event.is_directory:
            index.update_file(event.src_path)
            threading.Thread(target=semantic.update_file, args=(event.src_path,), daemon=True).start()


def _is_natural_language(query: str) -> bool:
    """Heuristic: multi-word prose queries go to semantic; identifiers/symbols go to pattern."""
    words = query.strip().split()
    if len(words) <= 1:
        return False
    # If every word is purely alphabetic (no underscores, dots, parens) it reads as prose
    return all(w.isalpha() for w in words)


_MAX_FILES = 10
_MAX_MATCHES_PER_FILE = 3


def _format_pattern_results(results: list, stats: dict) -> str:
    if not results:
        return None
    total_files = len(results)
    total_matches = sum(len(r['matches']) for r in results)
    lines = [f"Found {total_matches} matches in {total_files} files ({stats['files']} files searched from VRAM):"]
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
        lines.append(f"\n... {total_files - _MAX_FILES} more files not shown")
    return "\n".join(lines)


def _format_semantic_results(results: list, query: str, s: dict) -> str:
    if not results:
        return None
    base = s["base_dir"]
    lines = [f"Top {len(results)} semantic matches for '{query}' ({s['chunks']} chunks searched):"]
    for r in results:
        rel = os.path.relpath(r["file"], base) if base else r["file"]
        lines.append(f"\n[score {r['score']}] {rel}  lines {r['start_line']}–{r['end_line']}")
        lines.append(r["snippet"])
    return "\n".join(lines)


@mcp.tool()
def search_code(query: str, top_k: int = 10) -> str:
    """
    Primary code search tool — use this for ALL searches. Auto-selects the best method:

    - Exact identifiers, function names, symbols, string literals, error messages
      → GPU pattern search (byte-exact, sub-millisecond)
    - Natural language: "where is X handled", "how does Y work", "authentication logic"
      → GPU semantic search (nomic-embed-code cosine similarity)

    Falls back to pattern search if the semantic index hasn't been built yet.
    Prefer this over calling gpu_search or gpu_semantic_search directly.
    """
    pattern_ready = index.stats()['files'] > 0
    semantic_ready = semantic.stats()['chunks'] > 0
    use_semantic = _is_natural_language(query) and semantic_ready

    if use_semantic:
        results = semantic.search(query, top_k=top_k)
        out = _format_semantic_results(results, query, semantic.stats())
        return out or f"No semantic matches for '{query}'"

    if not pattern_ready:
        return "No index found. Call gpu_index (and optionally gpu_semantic_index) with your project directory first."

    results = index.search(query)
    out = _format_pattern_results(results, index.stats())
    if not out:
        return f"No matches for '{query}'"

    # Append blast radius for matched files
    if deps.stats()["files"] > 0 and results:
        matched_files = list({r["file"] for r in results[:5]})
        all_impact: dict[str, int] = {}
        for f in matched_files:
            for item in deps.impact(f):
                if item["file"] not in all_impact or item["hops"] < all_impact[item["file"]]:
                    all_impact[item["file"]] = item["hops"]
        if all_impact:
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
def gpu_search(query: str, case_sensitive: bool = False) -> str:
    """
    Explicit exact-text search — use when you know the precise identifier, symbol, or literal.
    Prefer search_code for general use; use this when case_sensitive control is needed.
    """
    stats = index.stats()
    if stats['files'] == 0:
        return "No files indexed. Call gpu_index with your project directory first."

    results = index.search(query, case_sensitive=case_sensitive)
    out = _format_pattern_results(results, stats)
    return out or f"No matches for '{query}'"


@mcp.tool()
def gpu_index(directory: str, append: bool = False) -> str:
    """
    Load a project directory into GPU VRAM for fast searching.
    Returns immediately — indexing runs in the background for large repos.
    Call gpu_stats to check when it's done.
    Set append=True to add a second root (multi-root support).
    """
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
    lines = [
        f"Pattern index:  {p['files']} files, {p['vram_mb']} MB  ({p['base_dir'] or 'none'})",
        f"Semantic index: {s['chunks']} chunks, {s['vram_mb']} MB  ({s['base_dir'] or 'not built'})",
        f"Dep graph:      {d['files']} files, {d['edges']} edges  ({d['base_dir'] or 'not built'})",
    ]
    if _bg_status["pattern"]:
        lines.append(f"Pattern status: {_bg_status['pattern']}")
    if _bg_status["deps"]:
        lines.append(f"Deps status:    {_bg_status['deps']}")
    return "\n".join(lines)


@mcp.tool()
def gpu_update_file(filepath: str) -> str:
    """Re-index a specific file after editing it (keeps VRAM in sync)."""
    index.update_file(filepath)
    return f"Updated: {filepath}"


@mcp.tool()
def gpu_semantic_index(directory: str) -> str:
    """
    Embed a project directory with bge-small-en-v1.5 and store vectors in GPU VRAM.
    Run once after server start — takes ~30s (model loads from local cache).
    Required before gpu_semantic_search or natural-language search_code queries.
    """
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    print(f"[gpu-search] Building semantic index for {directory}...", file=sys.stderr, flush=True)
    stats = semantic.index_directory(directory)
    print(f"[gpu-search] Semantic index ready: {stats['chunks']} chunks ({stats['vram_mb']} MB VRAM)", file=sys.stderr, flush=True)
    return (
        f"Semantic index built: {stats['chunks']} chunks embedded into VRAM "
        f"({stats['vram_mb']} MB). Skipped {stats['skipped']} files."
    )


@mcp.tool()
def dep_impact(filepath: str) -> str:
    """
    CALL THIS BEFORE EDITING ANY FILE.
    Uses GPU BFS to find every file that transitively imports the given file —
    i.e., everything that could break if you change it.
    Returns files grouped by hop distance (direct importers first).
    Runs in milliseconds from the in-VRAM dependency graph.
    """
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

    lines = [f"Impact of changing '{os.path.relpath(filepath, base)}' ({len(results)} affected files):"]
    for hop in sorted(by_hop):
        label = "Direct importers" if hop == 1 else f"Indirect (depth {hop})"
        lines.append(f"\n{label}:")
        for f in by_hop[hop]:
            lines.append(f"  {os.path.relpath(f, base)}")
    return "\n".join(lines)


@mcp.tool()
def dep_imports(filepath: str) -> str:
    """
    Show every project file that the given file directly imports.
    Useful for understanding a file's direct dependencies before refactoring.
    """
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
def dep_index(directory: str, append: bool = False) -> str:
    """
    Build the sparse GPU dependency graph for a project directory.
    Parses imports/requires across Python, JS, TS, Go, Rust, Java, C#, Ruby.
    Returns immediately — runs in background for large repos. Call gpu_stats to check progress.
    Set append=True to add a second root (multi-root / monorepo support).
    Required before dep_impact or dep_imports.
    """
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"

    def _do():
        _bg_status["deps"] = f"indexing {directory}..."
        s = deps.index_directory(directory, append=append)
        _bg_status["deps"] = f"done: {s['files']} files, {s['edges']} edges"

    threading.Thread(target=_do, daemon=True).start()
    return f"Dep graph indexing started for {directory} — call gpu_stats to check progress."


@mcp.tool()
def gpu_semantic_search(query: str, top_k: int = 10) -> str:
    """
    Search the codebase by meaning using GPU cosine similarity.
    Finds relevant code even without knowing exact function/variable names.
    Call gpu_semantic_index first to build the embedding index.
    """
    s = semantic.stats()
    if s["chunks"] == 0:
        return "No semantic index found. Call gpu_semantic_index with your project directory first."

    results = semantic.search(query, top_k=top_k)
    if not results:
        return f"No results for '{query}'"

    base = s["base_dir"]
    lines = [f"Top {len(results)} semantic matches for '{query}' (searched {s['chunks']} chunks):"]
    for r in results:
        rel = os.path.relpath(r["file"], base) if base else r["file"]
        lines.append(f"\n[score {r['score']}] {rel}  lines {r['start_line']}–{r['end_line']}")
        lines.append(r["snippet"])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", action="append", dest="directories",
                        metavar="DIR", help="Directory to index (repeat for multi-root)")
    args = parser.parse_args()

    targets = [os.path.abspath(d) for d in (args.directories or [os.getcwd()])]
    targets = [t for t in targets if os.path.isdir(t)]

    observer = Observer()

    for i, target in enumerate(targets):
        observer.schedule(_Watcher(), target, recursive=True)

    def _startup_index():
        for i, target in enumerate(targets):
            append = i > 0
            _bg_status["pattern"] = f"indexing {target}..."
            stats = index.index_directory(target, append=append)
            _bg_status["pattern"] = f"done: {stats['indexed']} files ({stats['vram_mb']} MB)"
            print(f"[gpu-search] Pattern index: {stats['indexed']} files ({stats['vram_mb']} MB VRAM) from {target}", file=sys.stderr)
            _bg_status["deps"] = f"indexing {target}..."
            dep_stats = deps.index_directory(target, append=append)
            _bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"
            print(f"[gpu-search] Dep graph: {dep_stats['files']} files, {dep_stats['edges']} edges", file=sys.stderr)

    threading.Thread(target=_startup_index, daemon=True).start()

    if targets:
        primary = targets[0]

        def _startup_semantic():
            s = semantic.try_load_cache(primary)
            if s:
                print(f"[gpu-search] Semantic cache loaded: {s['chunks']} chunks ({s['vram_mb']} MB VRAM)", file=sys.stderr, flush=True)
                semantic._get_model()
                print(f"[gpu-search] Semantic model ready", file=sys.stderr, flush=True)

        threading.Thread(target=_startup_semantic, daemon=True).start()
        observer.daemon = True
        observer.start()

    mcp.run(transport="stdio")
