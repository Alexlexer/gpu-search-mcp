#!/usr/bin/env python3
"""
gpu-search-mcp benchmark: compares gpu-search pattern search vs ripgrep.

Usage:
  python benchmarks/run_benchmark.py --repo /path/to/large/repo [--runs 10]

Outputs a Markdown table with median and p95 latency.
Requires ripgrep (rg) on PATH for the comparison columns.
Run on a warm machine (indexing complete) for valid comparisons.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

QUERIES = [
    "handleError",
    "createTextModel",
    "addEventListener",
    "disposeOnReturn",
    "ICodeEditor",
]

_RG = shutil.which("rg") or shutil.which("ripgrep")
_CARGO = shutil.which("cargo")


def _time_fn(fn, runs: int) -> list[float]:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _median_p95(times: list[float]) -> tuple[float, float]:
    s = sorted(times)
    med = statistics.median(s)
    p95 = s[int(len(s) * 0.95)]
    return med, p95


def _rg_search(query: str, repo: str, cold: bool = False) -> None:
    cmd = [_RG, "--count-matches", "-r", "", query, repo]
    if cold:
        # Attempt to drop OS file cache (Linux only; requires root)
        try:
            subprocess.run(
                ["sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
                check=False, capture_output=True,
            )
        except Exception:
            pass
    subprocess.run(cmd, capture_output=True, text=True)


def _repo_info(repo: str) -> dict:
    n_files = sum(1 for _ in Path(repo).rglob("*") if Path(_).is_file())
    total_mb = sum(
        f.stat().st_size for f in Path(repo).rglob("*") if f.is_file()
    ) / 1024 / 1024
    return {"files": n_files, "size_mb": round(total_mb, 1)}


def _rust_core_benchmark(repo: str, runs: int, queries: list[str]) -> dict | None:
    if not _CARGO:
        return None

    cmd = [
        _CARGO,
        "run",
        "--release",
        "--quiet",
        "--example",
        "pattern_benchmark",
        "--manifest-path",
        str(REPO_ROOT / "crates" / "gpu-search-core" / "Cargo.toml"),
        "--",
        "--repo",
        repo,
        "--runs",
        str(runs),
    ]
    for query in queries:
        cmd.extend(["--query", query])

    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print("Rust core benchmark failed; skipping column.", file=sys.stderr)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        return None
    return json.loads(completed.stdout)


def main():
    parser = argparse.ArgumentParser(description="Benchmark gpu-search vs ripgrep")
    parser.add_argument("--repo", default=str(REPO_ROOT), help="Repository to benchmark against")
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs per query")
    parser.add_argument("--no-rg", action="store_true", help="Skip ripgrep comparison")
    parser.add_argument(
        "--rust-core",
        action="store_true",
        help="Also benchmark the experimental Rust core pattern-search prototype",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        sys.exit(f"Not a directory: {repo}")

    print(f"Benchmarking against: {repo}")
    print(f"Runs per query: {args.runs}")
    print()

    # Build gpu-search index
    from gpu_index import GpuFileIndex
    import torch

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    info = _repo_info(repo)
    print(f"Repo: {info['files']} files, {info['size_mb']} MB")
    print()

    rust_results_by_query = {}
    if args.rust_core:
        print("Running Rust core pattern benchmark...")
        rust_output = _rust_core_benchmark(repo, args.runs, QUERIES)
        if rust_output:
            rust_results_by_query = {
                row["query"]: row for row in rust_output.get("results", [])
            }
            print(
                "  Rust core indexed "
                f"{rust_output.get('indexed_files')} files in "
                f"{rust_output.get('index_build_ms')}ms"
            )
        elif not _CARGO:
            print("  cargo not found on PATH — skipping Rust core comparison")
        print()

    print("Building pattern index...")
    t0 = time.perf_counter()
    idx = GpuFileIndex()
    stats = idx.index_directory(repo)
    index_time = time.perf_counter() - t0
    print(f"  Indexed {stats['indexed']} files ({stats['vram_mb']} MB VRAM) in {index_time:.1f}s")
    print()

    rows = []
    for query in QUERIES:
        gpu_times = _time_fn(lambda q=query: idx.search(q), args.runs)
        gpu_med, gpu_p95 = _median_p95(gpu_times)

        rg_warm_med = rg_warm_p95 = None
        if not args.no_rg and _RG:
            rg_times = _time_fn(lambda q=query: _rg_search(q, repo), args.runs)
            rg_warm_med, rg_warm_p95 = _median_p95(rg_times)

        matches = len(idx.search(query))
        rust_result = rust_results_by_query.get(query)
        rows.append((query, matches, gpu_med, gpu_p95, rg_warm_med, rg_warm_p95, rust_result))

    # Print results table
    print("## Results\n")
    print(f"Hardware: {device.upper()}  |  Repo: {info['files']} files, {info['size_mb']} MB  |  Runs: {args.runs}")
    print()
    if _RG and not args.no_rg:
        rust_headers = " Rust core median | Rust core p95 |" if args.rust_core else ""
        rust_sep = "------------------|---------------|" if args.rust_core else ""
        print(f"| Query | Matches | gpu-search median | gpu-search p95 | rg warm median | rg warm p95 |{rust_headers}")
        print(f"|-------|---------|-------------------|----------------|----------------|-------------|{rust_sep}")
        for q, m, gm, gp, rm, rp, rust_result in rows:
            rg_m = f"{rm*1000:.1f}ms" if rm is not None else "n/a"
            rg_p = f"{rp*1000:.1f}ms" if rp is not None else "n/a"
            rust_cols = ""
            if args.rust_core:
                rust_cols = (
                    f" {rust_result['p50_ms']:.1f}ms | {rust_result['p95_ms']:.1f}ms |"
                    if rust_result
                    else " n/a | n/a |"
                )
            print(f"| `{q}` | {m} | **{gm*1000:.1f}ms** | {gp*1000:.1f}ms | {rg_m} | {rg_p} |{rust_cols}")
    else:
        rust_headers = " Rust core median | Rust core p95 |" if args.rust_core else ""
        rust_sep = "------------------|---------------|" if args.rust_core else ""
        print(f"| Query | Matches | gpu-search median | gpu-search p95 |{rust_headers}")
        print(f"|-------|---------|-------------------|----------------|{rust_sep}")
        for q, m, gm, gp, *_rest in rows:
            rust_result = _rest[-1]
            rust_cols = ""
            if args.rust_core:
                rust_cols = (
                    f" {rust_result['p50_ms']:.1f}ms | {rust_result['p95_ms']:.1f}ms |"
                    if rust_result
                    else " n/a | n/a |"
                )
            print(f"| `{q}` | {m} | **{gm*1000:.1f}ms** | {gp*1000:.1f}ms |{rust_cols}")

    if not _RG:
        print("\n(ripgrep not found on PATH — skipping rg comparison)")

    print("\n### Methodology")
    print("See benchmarks/methodology.md for full details.")


if __name__ == "__main__":
    main()
