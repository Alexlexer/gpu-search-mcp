"""HTTP request handler and routing for gpu-search-mcp.

Imports mcp_server as _app for shared state (index, semantic, deps, etc.).
This is a deliberate circular import: mcp_server defines all global state
before importing this module, so the partial module reference is complete
enough by the time any handler method is called.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from ast_expand import read_block, skeleton_file
from redact import redact

# Late circular reference — mcp_server triggers this import only after
# all global state (index, semantic, deps, helpers) is fully defined.
import mcp_server as _app


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------

def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _get_device_dict() -> dict:
    """Return device metadata for /health and /stats responses."""
    try:
        from gpu_index import DEVICE_INFO
        return DEVICE_INFO.as_dict()
    except Exception:
        return {"backend": "unknown", "torchDevice": "unknown",
                "reason": "Device info unavailable", "warnings": []}


def _active_roots() -> list[str]:
    roots = list(_app._http_roots)
    for stats in (_app.index.stats(), _app.semantic.stats(), _app.deps.stats()):
        base = stats.get("base_dir")
        if base:
            roots.append(base)
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


def _is_allowed_result(abs_path: str, active_roots: list[str]) -> bool:
    """Return True iff abs_path is under an active root and not under .gpu-search-cache.

    Defense-in-depth: even if the in-memory index carries paths from a previous
    session or another repository, this prevents those paths from appearing in
    HTTP responses.
    """
    if not abs_path:
        return True
    try:
        p = Path(abs_path).resolve()
    except Exception:
        return False
    if ".gpu-search-cache" in p.parts:
        return False
    if not active_roots:
        return True
    resolved_roots: list[Path] = []
    for r in active_roots:
        try:
            resolved_roots.append(Path(r).resolve())
        except Exception:
            pass
    return any(p == root or p.is_relative_to(root) for root in resolved_roots)


def _filter_to_active_roots(results: list[dict], active_roots: list[str]) -> list[dict]:
    """Remove result entries whose absoluteFile is outside active roots or under .gpu-search-cache."""
    return [
        r for r in results
        if _is_allowed_result(r.get("absoluteFile") or r.get("file", ""), active_roots)
    ]


def _run_signal(signal: dict, top_k: int, context_mode: str) -> list[dict]:
    """Run all queries for one signal; return deduplicated matches capped at top_k."""
    base = _app.index.stats().get("base_dir") or ""
    active_roots = _active_roots()
    seen: set[tuple[str, int]] = set()
    matches: list[dict] = []
    snippet_limit = 160 if context_mode == "compact" else 300

    for query in signal["queries"]:
        if len(matches) >= top_k:
            break
        results = _app.index.search(query)
        for r in results:
            if len(matches) >= top_k:
                break
            file_abs = r["file"]
            if not _is_allowed_result(file_abs, active_roots):
                continue
            rel = os.path.relpath(file_abs, base) if base else file_abs
            for m in r.get("matches", []):
                if len(matches) >= top_k:
                    break
                key = (file_abs, m["line"])
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "file": rel,
                    "absoluteFile": file_abs,
                    "lineStart": m["line"],
                    "lineEnd": m["line"],
                    "score": 1.0,
                    "reason": f"pattern match: {query}",
                    "snippet": redact(m.get("content", ""))[:snippet_limit],
                    "engine": "pattern",
                })
    return matches


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

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
            return _json_response(self, 200, {
                "ok": True,
                "version": _app.VERSION,
                "device": _get_device_dict(),
            })
        if path == "/semantic/model/status":
            return _json_response(self, 200, _app.semantic_model_status_for_stats())
        if path == "/diagnostics":
            return _json_response(self, 200, _app.diagnostics_snapshot())
        if path == "/stats":
            p_stats = _app.index.stats()
            s_stats = _app.semantic.stats()
            d_stats = _app.deps.stats()
            return _json_response(self, 200, {
                "pattern": p_stats,
                "semantic": s_stats,
                "dependency": d_stats,
                "status": _app._bg_status,
                "capabilities": {
                    "patternSearch": p_stats["files"] > 0,
                    "semanticSearch": s_stats["chunks"] > 0,
                    "dependencyImpact": d_stats["files"] > 0,
                    "csharpAst": _csharp_ast_available(),
                    "httpStructuredResponses": True,
                },
                "limitations": _app._GLOBAL_LIMITATIONS,
                "device": _get_device_dict(),
                "cache": _app.cache_metadata_for_stats(),
                "semanticModel": _app.semantic_model_status_for_stats(),
            })
        if path == "/index/status":
            p_stats = _app.index.stats()
            s_stats = _app.semantic.stats()
            d_stats = _app.deps.stats()
            return _json_response(self, 200, {
                "indexedRoots": _active_roots(),
                "pattern": {
                    "ready": int(p_stats.get("files") or 0) > 0,
                    "files": p_stats.get("files", 0),
                    "baseDir": p_stats.get("base_dir"),
                    "cacheStatus": p_stats.get("cache"),
                },
                "dependency": {
                    "ready": int(d_stats.get("files") or 0) > 0,
                    "files": d_stats.get("files", 0),
                    "edges": d_stats.get("edges", 0),
                },
                "semantic": {
                    "ready": int(s_stats.get("chunks") or 0) > 0,
                    "chunks": s_stats.get("chunks", 0),
                    "baseDir": s_stats.get("base_dir"),
                },
                "status": _app._bg_status,
                "lastIndexResult": _app._last_index_result,
            })
        return _json_response(self, 404, {"error": "not found"})

    def do_POST(self):
        try:
            payload = self._read_json()
            path = urlparse(self.path).path

            if path == "/search/code":
                mode = payload.get("mode", "auto")
                context_mode = payload.get("contextMode", payload.get("context_mode", "normal"))
                result = _app.search_code(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode=mode,
                    context_mode=context_mode,
                )
                structured = _app._http_search_structured(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode=mode,
                    context_mode=context_mode,
                )
                structured["results"] = _filter_to_active_roots(structured.get("results", []), _active_roots())
                return _json_response(self, 200, {"result": result, **structured})

            if path == "/search/hybrid":
                context_mode = payload.get("contextMode", payload.get("context_mode", "normal"))
                result = _app.search_code(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="hybrid",
                    context_mode=context_mode,
                )
                structured = _app._http_search_structured(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="hybrid",
                    context_mode=context_mode,
                )
                structured["results"] = _filter_to_active_roots(structured.get("results", []), _active_roots())
                return _json_response(self, 200, {"result": result, **structured})

            if path == "/search/semantic":
                context_mode = payload.get("contextMode", payload.get("context_mode", "normal"))
                result = _app.search_code(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="semantic",
                    context_mode=context_mode,
                )
                structured = _app._http_search_structured(
                    payload.get("query", ""),
                    top_k=int(payload.get("topK", payload.get("top_k", 5))),
                    mode="semantic",
                    context_mode=context_mode,
                )
                structured["results"] = _filter_to_active_roots(structured.get("results", []), _active_roots())
                return _json_response(self, 200, {"result": result, **structured})

            if path == "/read/block":
                safe_path = _require_under_root(payload.get("filepath", payload.get("file", "")))
                line = int(payload.get("line", 1))
                if not os.path.isfile(safe_path):
                    return _json_response(self, 400, {"error": f"File not found: {safe_path}"})
                code, line_start, line_end = read_block(safe_path, line)
                base = _app.index.stats().get("base_dir") or os.path.dirname(safe_path)
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
                base = _app.index.stats().get("base_dir") or os.path.dirname(safe_path)
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
                result = _app.dep_impact(safe_path)
                dep_stats = _app.deps.stats()
                base = dep_stats.get("base_dir") or ""
                rel = os.path.relpath(safe_path, base) if base else safe_path
                warnings: list[str] = []
                if dep_stats["files"] > 0:
                    confidence = "medium"
                    try:
                        impact_list = _app.deps.impact(safe_path)
                        active_roots = _active_roots()
                        impacted_files = [
                            {
                                "file": os.path.relpath(r["file"], base) if base else r["file"],
                                "absoluteFile": r["file"],
                                "hops": r["hops"],
                                **({"reason": r["reason"]} if r.get("reason") else {}),
                            }
                            for r in impact_list
                            if _is_allowed_result(r["file"], active_roots)
                        ]
                    except Exception:
                        impacted_files = []
                    if not impacted_files:
                        warnings.append("No files in the dependency graph import this path.")
                else:
                    confidence = "low"
                    impacted_files = []
                    warnings.append("Dependency graph not built. Call dep_index first to build it.")
                return _json_response(self, 200, {
                    "result": result,
                    "file": rel,
                    "absoluteFile": safe_path,
                    "confidence": confidence,
                    "analysisMode": "heuristic",
                    "limitations": _app._DEP_LIMITATIONS,
                    "warnings": warnings,
                    "impactedFiles": impacted_files,
                })

            if path == "/index/root":
                directory = payload.get("directory", "")
                if not directory:
                    return _json_response(self, 400, {"error": "directory is required"})
                directory = os.path.abspath(directory)
                if not os.path.exists(directory):
                    return _json_response(self, 400, {"error": f"Directory not found: {directory}"})
                if not os.path.isdir(directory):
                    return _json_response(self, 400, {"error": f"Not a directory: {directory}"})
                rebuild_cache = bool(payload.get("rebuildCache", False))
                include_semantic = bool(payload.get("includeSemantic", False))
                result = _app._index_root(directory, rebuild_cache=rebuild_cache, include_semantic=include_semantic)
                return _json_response(self, 200 if result["ok"] else 500, result)

            if path == "/scan/signals":
                categories_filter = payload.get("categories")
                top_k = min(int(payload.get("topKPerSignal", 5)), 20)
                include_snippets = bool(payload.get("includeSnippets", True))
                context_mode = payload.get("contextMode", "compact")

                if _app.index.stats()["files"] == 0:
                    return _json_response(self, 200, {
                        "result": "No pattern index found. Call gpu_index first.",
                        "categories": [],
                        "summary": {"signalCount": 0, "matchCount": 0, "categories": {}},
                        "signals": [],
                        "limitations": _app._SIGNAL_SCAN_LIMITATIONS,
                        "warnings": ["Pattern index not built. Call gpu_index first."],
                    })

                signals_to_run = _app._BUILTIN_SIGNALS
                if categories_filter:
                    signals_to_run = [s for s in _app._BUILTIN_SIGNALS if s["category"] in categories_filter]

                _MAX_TOTAL_MATCHES = 200
                result_signals: list[dict] = []
                scan_warnings: list[str] = []
                total_so_far = 0

                for signal in signals_to_run:
                    if total_so_far >= _MAX_TOTAL_MATCHES:
                        scan_warnings.append(
                            f"Total match cap ({_MAX_TOTAL_MATCHES}) reached — remaining signals skipped."
                        )
                        break
                    try:
                        matches = _run_signal(signal, top_k, context_mode)
                        remaining_cap = _MAX_TOTAL_MATCHES - total_so_far
                        matches = matches[:remaining_cap]
                        if not include_snippets:
                            for m in matches:
                                m.pop("snippet", None)
                        total_so_far += len(matches)
                        result_signals.append({
                            "id": signal["id"],
                            "category": signal["category"],
                            "label": signal["label"],
                            "description": signal["description"],
                            "confidence": signal["confidence"],
                            "query": " OR ".join(signal["queries"]),
                            "matches": matches,
                        })
                    except Exception as exc:
                        scan_warnings.append(f"Signal '{signal['id']}' failed: {exc}")

                cats_seen = list(dict.fromkeys(s["category"] for s in signals_to_run))
                cats_summary: dict[str, int] = {}
                signal_count = 0
                match_count = 0
                for sig in result_signals:
                    n = len(sig["matches"])
                    if n:
                        signal_count += 1
                        match_count += n
                        cats_summary[sig["category"]] = cats_summary.get(sig["category"], 0) + n

                result_str = (
                    f"Signal scan: {signal_count} signal{'s' if signal_count != 1 else ''} with matches, "
                    f"{match_count} total match{'es' if match_count != 1 else ''} "
                    f"across {len(cats_seen)} categor{'ies' if len(cats_seen) != 1 else 'y'}."
                )
                return _json_response(self, 200, {
                    "result": result_str,
                    "categories": cats_seen,
                    "summary": {
                        "signalCount": signal_count,
                        "matchCount": match_count,
                        "categories": cats_summary,
                    },
                    "signals": result_signals,
                    "limitations": _app._SIGNAL_SCAN_LIMITATIONS,
                    "warnings": scan_warnings,
                })

            return _json_response(self, 404, {"error": "not found"})
        except ValueError as e:
            return _json_response(self, 400, {"error": str(e)})
        except Exception as e:
            return _json_response(self, 500, {"error": str(e)})
