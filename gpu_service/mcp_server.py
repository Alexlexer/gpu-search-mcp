"""
GPU-accelerated codebase search MCP server.
Files are pre-loaded into RTX VRAM; searches run as parallel CUDA kernels.

Usage: python mcp_server.py [--directory PATH]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from mcp.server.fastmcp import FastMCP
from gpu_index import GpuFileIndex, INDEXED_EXTS
from gpu_semantic_index import SemanticIndex
from pathlib import Path

mcp = FastMCP("gpu-search")
index = GpuFileIndex()
semantic = SemanticIndex()


class _Watcher(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and Path(event.src_path).suffix.lower() in INDEXED_EXTS:
            index.update_file(event.src_path)

    def on_created(self, event):
        self.on_modified(event)

    def on_deleted(self, event):
        if not event.is_directory:
            index.update_file(event.src_path)


def _is_natural_language(query: str) -> bool:
    """Heuristic: multi-word prose queries go to semantic; identifiers/symbols go to pattern."""
    words = query.strip().split()
    if len(words) <= 1:
        return False
    # If every word is purely alphabetic (no underscores, dots, parens) it reads as prose
    return all(w.isalpha() for w in words)


def _format_pattern_results(results: list, stats: dict) -> str:
    if not results:
        return None
    lines = [f"Found matches in {len(results)} files ({stats['files']} files searched from VRAM):"]
    for r in results[:20]:
        rel = os.path.relpath(r['file'], stats['base_dir']) if stats['base_dir'] else r['file']
        lines.append(f"\n{rel}:")
        for m in r['matches']:
            lines.append(f"  {m['line']}: {m['content']}")
    total = sum(len(r['matches']) for r in results)
    lines.append(f"\n--- {total} total matches ---")
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
    return out or f"No matches for '{query}'"


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
def gpu_index(directory: str) -> str:
    """Load a project directory into GPU VRAM for fast searching. Re-call to refresh."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    stats = index.index_directory(directory)
    return (
        f"Indexed {stats['indexed']} files into VRAM "
        f"({stats['vram_mb']} MB used). "
        f"Skipped {stats['skipped']} files."
    )


@mcp.tool()
def gpu_stats() -> str:
    """Show VRAM usage for both the pattern index and the semantic embedding index."""
    p = index.stats()
    s = semantic.stats()
    return (
        f"Pattern index:  {p['files']} files, {p['vram_mb']} MB VRAM  ({p['base_dir'] or 'none'})\n"
        f"Semantic index: {s['chunks']} chunks, {s['vram_mb']} MB VRAM  ({s['base_dir'] or 'not built'})"
    )


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
    parser.add_argument("--directory", "-d", help="Directory to index (defaults to cwd)")
    args = parser.parse_args()

    # Auto-index: use --directory if given, otherwise the current working directory
    target = os.path.abspath(args.directory) if args.directory else os.getcwd()

    if os.path.isdir(target):
        stats = index.index_directory(target)
        print(
            f"[gpu-search] Pattern index: {stats['indexed']} files "
            f"({stats['vram_mb']} MB VRAM) from {target}",
            file=sys.stderr,
        )

        observer = Observer()
        observer.schedule(_Watcher(), target, recursive=True)
        observer.daemon = True
        observer.start()

    mcp.run(transport="stdio")
