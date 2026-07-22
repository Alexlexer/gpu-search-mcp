"""Benchmark CLI for gpu-search-mcp.

Examples:
  gpu-search-bench --directory D:\repos\vscode --queries benchmarks/queries.json --output results.json
  gpu-search-bench --directory benchmarks/fixtures/csharp --manifest benchmarks/manifests/csharp.json --output quality.json
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


_QUALITY_MODES = {
    "ripgrep",
    "exact",
    "symbol",
    "semantic",
    "hybrid",
    "hybrid_symbols",
    "hybrid_dependencies",
}


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
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "node_modules", ".venv", "venv", "dist", "build"}
        ]
        for name in names:
            try:
                total_bytes += os.path.getsize(os.path.join(root, name))
                files += 1
            except OSError:
                pass
    return {
        "files": files,
        "bytes": total_bytes,
        "mb": round(total_bytes / 1024 / 1024, 2),
    }


def _ripgrep_latency(directory: str, query: str) -> float | None:
    try:
        t0 = time.perf_counter()
        subprocess.run(
            ["rg", "--fixed-strings", "--", query, directory],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return (time.perf_counter() - t0) * 1000
    except FileNotFoundError:
        return None


def _ripgrep_results(directory: str, query: str, top_k: int) -> dict:
    try:
        completed = subprocess.run(
            ["rg", "--files-with-matches", "--fixed-strings", "--", query, directory],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return {
            "results": [],
            "primary_results": [],
            "related_files": {"tests": []},
            "warnings": [
                "ripgrep is not installed; baseline mode returned no results."
            ],
        }
    results = [
        {"file": path, "absoluteFile": path, "engine": "ripgrep"}
        for path in completed.stdout.splitlines()[:top_k]
    ]
    return {
        "results": results,
        "primary_results": results,
        "related_files": {"tests": []},
        "warnings": [],
    }


def _cache_size(directory: str) -> int:
    cache = Path(directory) / ".gpu-search-cache"
    if not cache.is_dir():
        return 0
    total = 0
    for path in cache.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            pass
    return total


def _peak_rss_bytes() -> int | None:
    if os.name == "nt":
        return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except (ImportError, OSError):
        return None


def _merge_unique_results(
    first: list[dict], second: list[dict], top_k: int
) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()
    for item in [*first, *second]:
        path = str(item.get("absoluteFile") or item.get("file") or "").casefold()
        if path and path not in seen:
            seen.add(path)
            merged.append(item)
        if len(merged) >= top_k:
            break
    return merged


def run_benchmark(
    directory: str, queries: list[str], iterations: int = 20
) -> dict:
    from .gpu_index import DEVICE_INFO, GpuFileIndex

    directory = os.path.abspath(directory)
    idx = GpuFileIndex()
    build_t0 = time.perf_counter()
    index_stats = idx.index_directory(directory)
    build_ms = (time.perf_counter() - build_t0) * 1000

    query_results = []
    for query in queries:
        idx.search(query, max_files=10)
        direct = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            matches = idx.search(query, max_files=10)
            direct.append((time.perf_counter() - t0) * 1000)
        rg_warm = _ripgrep_latency(directory, query)
        query_results.append({
            "query": query,
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
            "device": DEVICE_INFO.torch_device,
            "deviceReason": DEVICE_INFO.reason,
            "deviceWarnings": DEVICE_INFO.warnings,
        },
        "repo": _repo_info(directory),
        "index": {
            **index_stats,
            "build_ms": round(build_ms, 3),
        },
        "methodology": {
            "iterations": iterations,
            "direct_python": "GpuFileIndex.search after one warm-up call",
            "ripgrep_warm": (
                "single rg --fixed-strings call after indexing; cold cache is "
                "OS-dependent and not forced"
            ),
            "mcp_overhead": (
                "measure separately by calling search_code through an MCP/HTTP client"
            ),
        },
        "queries": query_results,
    }


def run_quality_manifest(
    directory: str,
    manifest_path: str,
    *,
    modes: list[str] | None = None,
    iterations: int = 1,
    top_k: int = 10,
    build_semantic: bool = False,
) -> dict:
    from . import mcp_server as app
    from .gpu_index import DEVICE_INFO
    from .quality_benchmark import load_manifest, run_quality_benchmark

    manifest = load_manifest(manifest_path)
    selected_modes = tuple(modes or manifest.modes)
    unknown = sorted(set(selected_modes).difference(_QUALITY_MODES))
    if unknown:
        raise ValueError(f"unknown quality benchmark modes: {', '.join(unknown)}")
    directory = os.path.abspath(directory)

    started = time.perf_counter()
    index_stats = app.index.index_directory(directory)
    dependency_stats = app.deps.index_directory(directory)
    symbol_stats = app.symbols.index_directory(directory)
    semantic_stats: dict = app.semantic.stats()
    needs_semantic = bool(
        {"semantic", "hybrid", "hybrid_symbols", "hybrid_dependencies"}
        .intersection(selected_modes)
    )
    if needs_semantic:
        loaded = app.semantic.try_load_cache(directory)
        if loaded is None and build_semantic:
            semantic_stats = app.semantic.index_directory(directory)
        else:
            semantic_stats = app.semantic.stats()
    index_ms = (time.perf_counter() - started) * 1000

    incremental_ms: float | None = None
    first_expected = manifest.queries[0].expected_files[0]
    incremental_path = os.path.join(directory, first_expected)
    if os.path.isfile(incremental_path):
        incremental_started = time.perf_counter()
        app.index.update_file(incremental_path)
        app.deps.update_file(incremental_path)
        app.symbols.update_file(incremental_path)
        incremental_ms = (time.perf_counter() - incremental_started) * 1000

    def search(query, mode: str, requested_top_k: int) -> dict:
        exact_query = query.exact_query or query.query
        symbol_query = query.symbol_query or exact_query
        if mode == "ripgrep":
            return _ripgrep_results(directory, exact_query, requested_top_k)
        if mode == "exact":
            return app._http_search_structured(
                exact_query, top_k=requested_top_k, mode="exact"
            )
        if mode == "symbol":
            return app._http_search_structured(
                symbol_query, top_k=requested_top_k, mode="symbol"
            )
        if mode == "semantic":
            return app._http_search_structured(
                query.query, top_k=requested_top_k, mode="semantic"
            )

        response = app._http_search_structured(
            query.query,
            top_k=requested_top_k,
            mode="hybrid",
            intent="understand",
            include_dependencies=mode == "hybrid_dependencies",
            include_tests=mode == "hybrid_dependencies",
        )
        if mode == "hybrid_symbols":
            symbol_response = app._http_search_structured(
                symbol_query, top_k=requested_top_k, mode="symbol"
            )
            results = _merge_unique_results(
                symbol_response.get("results", []),
                response.get("results", []),
                requested_top_k,
            )
            response = {
                **response,
                "results": results,
                "primary_results": results,
                "warnings": list(dict.fromkeys([
                    *response.get("warnings", []),
                    *symbol_response.get("warnings", []),
                ])),
            }
        return response

    files = int(index_stats.get("files") or index_stats.get("indexed") or 0)
    runtime = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": DEVICE_INFO.torch_device,
        "device_reason": DEVICE_INFO.reason,
        "device_warnings": DEVICE_INFO.warnings,
        "index_build_ms": round(index_ms, 3),
        "indexing_files_per_second": (
            round(files / (index_ms / 1000), 3) if index_ms > 0 else None
        ),
        "incremental_update_ms": (
            round(incremental_ms, 3) if incremental_ms is not None else None
        ),
        "peak_ram_bytes": _peak_rss_bytes(),
        "vram_mb": index_stats.get("vram_mb"),
        "cache_size_bytes": _cache_size(directory),
        "pattern_index": index_stats,
        "dependency_index": dependency_stats,
        "symbol_index": symbol_stats,
        "semantic_index": semantic_stats,
        "limitations": [
            (
                "peak_ram_bytes is unavailable on Windows without an optional "
                "process-metrics dependency."
            )
        ] if os.name == "nt" else [],
    }
    return run_quality_benchmark(
        manifest,
        search,
        modes=selected_modes,
        top_k=top_k,
        iterations=iterations,
        repository_root=directory,
        runtime=runtime,
    )


def _quality_regressions(result: dict, args) -> list[dict]:
    if not args.baseline:
        return []
    from .quality_benchmark import compare_baseline

    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    thresholds_selected = any(
        value is not None
        for value in (
            args.max_quality_drop,
            args.max_latency_increase_pct,
            args.max_token_increase_pct,
        )
    )
    if not thresholds_selected:
        return []
    return compare_baseline(
        result,
        baseline,
        max_quality_drop=(
            args.max_quality_drop
            if args.max_quality_drop is not None
            else float("inf")
        ),
        max_latency_increase_pct=args.max_latency_increase_pct,
        max_token_increase_pct=args.max_token_increase_pct,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark gpu-search-mcp")
    parser.add_argument("--directory", "-d", required=True)
    parser.add_argument("--queries", help="JSON list or {'queries': [...]} file")
    parser.add_argument(
        "--manifest",
        help="Quality manifest (JSON or YAML); enables retrieval-quality metrics",
    )
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Torch device to use (default: auto -> cuda > mps > cpu)",
    )
    parser.add_argument(
        "--modes",
        help="Comma-separated quality modes; defaults to the manifest modes",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--build-semantic",
        action="store_true",
        help="Build semantic embeddings when no compatible cache is available",
    )
    parser.add_argument("--baseline", help="Existing quality report to compare")
    parser.add_argument("--write-baseline", help="Write a copy as a future baseline")
    parser.add_argument("--max-quality-drop", type=float)
    parser.add_argument("--max-latency-increase-pct", type=float)
    parser.add_argument("--max-token-increase-pct", type=float)
    args = parser.parse_args(argv)

    os.environ["GPU_SEARCH_DEVICE"] = args.device
    if args.manifest:
        modes = [
            item.strip() for item in (args.modes or "").split(",") if item.strip()
        ] or None
        result = run_quality_manifest(
            args.directory,
            args.manifest,
            modes=modes,
            iterations=args.iterations,
            top_k=args.top_k,
            build_semantic=args.build_semantic,
        )
        regressions = _quality_regressions(result, args)
        if args.baseline:
            result["baseline_comparison"] = {
                "baseline": args.baseline,
                "thresholds": {
                    "max_quality_drop": args.max_quality_drop,
                    "max_latency_increase_pct": args.max_latency_increase_pct,
                    "max_token_increase_pct": args.max_token_increase_pct,
                },
                "regressions": regressions,
            }
    else:
        result = run_benchmark(
            args.directory, _load_queries(args.queries), args.iterations
        )
        regressions = []

    encoded = json.dumps(result, indent=2, sort_keys=True)
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    if args.write_baseline:
        Path(args.write_baseline).write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
