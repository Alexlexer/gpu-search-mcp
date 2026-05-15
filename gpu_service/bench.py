"""Benchmark CLI for gpu-search-mcp.

Example:
  gpu-search-bench --directory D:\repos\vscode --queries benchmarks/queries.json --output results.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

from .gpu_index import GpuFileIndex, DEVICE


def _load_queries(path: str | None) -> list[str]:
    if not path:
        return ["class", "function", "authentication", "TODO"]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x["query"] if isinstance(x, dict) else x) for x in data]
    return [str(x) for x in data.get("queries", [])]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = int(round((len(values) - 1) * pct))
    return values[k]


def _repo_info(directory: str) -> dict:
    total_bytes = 0
    files = 0
    for root, dirs, names in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in {".git", "node_modules", ".venv", "venv", "dist", "build"}]
        for name in names:
            try:
                total_bytes += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return {"files": files, "bytes": total_bytes, "mb": round(total_bytes / 1024 / 1024, 2)}


def _ripgrep_latency(directory: str, query: str) -> float | None:
    exe = "rg"
    try:
        t0 = time.perf_counter()
        subprocess.run([exe, "--fixed-strings", "--", query, directory],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return (time.perf_counter() - t0) * 1000
    except FileNotFoundError:
        return None


def run_benchmark(directory: str, queries: list[str], iterations: int = 20) -> dict:
    directory = os.path.abspath(directory)
    idx = GpuFileIndex()
    build_t0 = time.perf_counter()
    index_stats = idx.index_directory(directory)
    build_ms = (time.perf_counter() - build_t0) * 1000

    query_results = []
    for q in queries:
        # warm-up
        idx.search(q, max_files=10)
        direct = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            matches = idx.search(q, max_files=10)
            direct.append((time.perf_counter() - t0) * 1000)
        rg_warm = _ripgrep_latency(directory, q)
        query_results.append({
            "query": q,
            "matches": len(matches),
            "direct_python_ms": {
                "p50": round(statistics.median(direct), 3),
                "p95": round(_percentile(direct, 0.95), 3),
                "p99": round(_percentile(direct, 0.99), 3),
                "min": round(min(direct), 3),
            },
            "ripgrep_warm_ms": None if rg_warm is None else round(rg_warm, 3),
        })

    return {
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
            "device": DEVICE.type,
        },
        "repo": _repo_info(directory),
        "index": {
            **index_stats,
            "build_ms": round(build_ms, 3),
        },
        "methodology": {
            "iterations": iterations,
            "direct_python": "GpuFileIndex.search after one warm-up call",
            "ripgrep_warm": "single rg --fixed-strings call after indexing; cold cache is OS-dependent and not forced",
            "mcp_overhead": "measure separately by calling search_code through an MCP/HTTP client",
        },
        "queries": query_results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark gpu-search-mcp pattern search")
    parser.add_argument("--directory", "-d", required=True)
    parser.add_argument("--queries", help="JSON list or {'queries': [...]} file")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    result = run_benchmark(args.directory, _load_queries(args.queries), args.iterations)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
