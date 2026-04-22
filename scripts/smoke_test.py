#!/usr/bin/env python3
"""Local smoke test for gpu-search without MCP transport."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import mcp_server  # noqa: E402


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return ok


def skip(name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[SKIP] {name}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT), help="Repository root to index")
    parser.add_argument("--with-semantic", action="store_true", help="Attempt semantic indexing/search too")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    target_file = os.path.join(repo, "gpu_service", "mcp_server.py")

    all_ok = True

    pattern_stats = mcp_server.index.index_directory(repo)
    all_ok &= check("pattern index", pattern_stats["indexed"] > 0, str(pattern_stats))

    dep_stats = mcp_server.deps.index_directory(repo)
    all_ok &= check("dependency index", dep_stats["files"] > 0, str(dep_stats))

    pattern_out = mcp_server.search_code("search_code", mode="pattern")
    all_ok &= check("pattern search", isinstance(pattern_out, str) and "Pattern:" in pattern_out)

    impact_out = mcp_server.dep_impact(target_file)
    all_ok &= check("dep impact", isinstance(impact_out, str) and len(impact_out) > 0)

    block_out = mcp_server.gpu_read_block(target_file, 292)
    all_ok &= check("read block", isinstance(block_out, str) and "search_code" in block_out)

    semantic_fmt = mcp_server._format_semantic_results(
        [{
            "file": target_file,
            "start_line": 292,
            "end_line": 299,
            "score": 0.99,
            "snippet": 'def search_code(query: str, top_k: int = 5, mode: str = "auto", ctx: Context = None) -> str:',
        }],
        "search code router",
        {"base_dir": repo},
        expand=False,
    )
    all_ok &= check("semantic formatter", isinstance(semantic_fmt, str) and "Semantic:" in semantic_fmt)

    if args.with_semantic:
        try:
            semantic_stats = mcp_server.semantic.index_directory(repo, force=True)
            semantic_out = mcp_server.search_code("where is code search routed", mode="semantic", top_k=2)
            all_ok &= check(
                "semantic index/search",
                semantic_stats["chunks"] > 0 and isinstance(semantic_out, str) and len(semantic_out) > 0,
                str(semantic_stats),
            )
        except Exception as e:
            skip("semantic index/search", str(e))
    else:
        skip("semantic index/search", "use --with-semantic to try model loading and semantic queries")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
