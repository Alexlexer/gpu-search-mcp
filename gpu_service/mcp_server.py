"""
GPU-accelerated codebase search MCP server.
Files are pre-loaded into RTX VRAM; searches run as parallel CUDA kernels.

Usage: python mcp_server.py [--directory PATH]
"""
import argparse
import asyncio
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


@mcp.tool()
def gpu_search(query: str, case_sensitive: bool = False) -> str:
    """Search for text in the indexed codebase using GPU. Faster than grep for large projects."""
    stats = index.stats()
    if stats['files'] == 0:
        return "No files indexed. Call gpu_index with your project directory first."

    results = index.search(query, case_sensitive=case_sensitive)
    if not results:
        return f"No matches for '{query}'"

    lines = [f"Found matches in {len(results)} files (searched {stats['files']} files from VRAM):"]
    for r in results[:20]:
        rel = os.path.relpath(r['file'], stats['base_dir']) if stats['base_dir'] else r['file']
        lines.append(f"\n{rel}:")
        for m in r['matches']:
            lines.append(f"  {m['line']}: {m['content']}")

    total = sum(len(r['matches']) for r in results)
    lines.append(f"\n--- {total} total matches ---")
    return "\n".join(lines)


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
    """Show how many files are in GPU VRAM and how much memory is used."""
    s = index.stats()
    return (
        f"Files in VRAM: {s['files']}\n"
        f"VRAM used: {s['vram_mb']} MB\n"
        f"Indexed directory: {s['base_dir'] or 'none'}"
    )


@mcp.tool()
def gpu_update_file(filepath: str) -> str:
    """Re-index a specific file after editing it (keeps VRAM in sync)."""
    index.update_file(filepath)
    return f"Updated: {filepath}"


@mcp.tool()
def gpu_semantic_index(directory: str) -> str:
    """
    Embed a project directory with nomic-embed-code and store vectors in GPU VRAM.
    Slower than gpu_index (embedding takes time); run once per project.
    Required before calling gpu_semantic_search.
    """
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    stats = semantic.index_directory(directory)
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
            f"[gpu-search] Auto-indexed {stats['indexed']} files "
            f"({stats['vram_mb']} MB VRAM) from {target}",
            file=sys.stderr
        )
        observer = Observer()
        observer.schedule(_Watcher(), target, recursive=True)
        observer.daemon = True
        observer.start()

    mcp.run(transport="stdio")
