"""
GPU-accelerated codebase search MCP server — bootstrap and compatibility entrypoint.

Module layout:
  mcp_server.py    — global state, helpers, startup, CLI (this file)
  server_config.py — constants and built-in signal definitions
  mcp_tools.py     — MCP tool/prompt registrations
  http_server.py   — HTTP request handler and routing

Usage: python mcp_server.py [--directory PATH]
"""
import argparse
import json
import platform
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection, HTTPException
from http.server import ThreadingHTTPServer
from pathlib import Path


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

sys.path.insert(0, os.path.dirname(__file__))

# Register this module under the bare `mcp_server` name immediately after
# adding gpu_service/ to sys.path.  Internal modules (mcp_tools, http_server)
# do `import mcp_server` which — without this — creates a second module
# instance with empty global state (different index/semantic/deps objects).
# Must run before any internal gpu_service import so the alias is present
# when those modules first call `import mcp_server`.
sys.modules.setdefault(
    "mcp_server",
    sys.modules.get("gpu_service.mcp_server") or sys.modules.get(__name__),
)

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from mcp.server.fastmcp import FastMCP
from ast_expand import read_block
from git_state import GitState
from redact import redact

from semantic_model_manager import (
    download_semantic_model,
    get_semantic_model_status,
    resolve_semantic_model_id,
    set_configured_semantic_model_id,
)
from server_config import (  # noqa: F401 (re-exported for mcp_server.* compatibility)
    VERSION,
    CONFIG_PATH,
    INDEXED_EXTS,
    SKIP_DIRS,
    _DEP_EXTS,
    MAX_CHUNKS,
    _DEP_LIMITATIONS,
    _GLOBAL_LIMITATIONS,
    _SIGNAL_SCAN_LIMITATIONS,
    _BUILTIN_SIGNALS,
    _load_config_dirs,
    _save_config_dirs,
)

# Set to True via --allow-env-files CLI flag or allow_env_files key in config JSON
_ALLOW_ENV_FILES: bool = False
_REBUILD_CACHE: bool = False
_SEMANTIC_MODEL_ID: str | None = None


# ---------------------------------------------------------------------------
# Lazy service wrappers
# ---------------------------------------------------------------------------

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


def cache_metadata_for_stats() -> dict:
    """Return additive cache metadata for HTTP /stats without changing service behavior."""
    from cache_manager import CACHE_SCHEMA_VERSION, cache_dir_for_repo, load_cache_metadata

    roots: list[str] = list(_http_roots)
    for service in (index, semantic, deps):
        try:
            base = service.stats().get("base_dir")
        except Exception:
            base = None
        if base:
            roots.append(base)
    root = next((r for r in roots if r), os.getcwd())
    cache_dir = cache_dir_for_repo(root)
    metadata = load_cache_metadata(cache_dir)
    return {
        "schemaVersion": CACHE_SCHEMA_VERSION,
        "directory": str(cache_dir),
        "entries": metadata.get("cacheEntries", []) if metadata else [],
    }


def semantic_model_status_for_stats() -> dict:
    """Return local-only semantic embedding model preflight status."""
    device = None
    gpu_index_module = sys.modules.get("gpu_index")
    if gpu_index_module is not None:
        try:
            device = gpu_index_module.DEVICE_INFO.torch_device
        except Exception:
            device = None
    return get_semantic_model_status(_SEMANTIC_MODEL_ID, device=device)


def diagnostics_snapshot() -> dict:
    """Return cheap local runtime diagnostics without indexing or downloading."""
    gpu_index_module = sys.modules.get("gpu_index")
    if gpu_index_module is not None:
        try:
            device = gpu_index_module.DEVICE_INFO.as_dict()
        except Exception:
            device = {
                "backend": "unknown",
                "torchDevice": "unknown",
                "reason": "Device info unavailable",
                "warnings": [],
            }
    else:
        device = {
            "backend": "unknown",
            "torchDevice": "unknown",
            "reason": "Device not initialized yet",
            "warnings": [],
        }

    warnings: list[str] = []
    try:
        p_stats = index.stats()
    except Exception as exc:
        p_stats = {"files": 0, "vram_mb": 0, "base_dir": None, "cache": "error"}
        warnings.append(f"Pattern index status unavailable: {exc}")
    try:
        s_stats = semantic.stats()
    except Exception as exc:
        s_stats = {"chunks": 0, "vram_mb": 0, "base_dir": None}
        warnings.append(f"Semantic index status unavailable: {exc}")
    try:
        d_stats = deps.stats()
    except Exception as exc:
        d_stats = {"files": 0, "edges": 0, "base_dir": None, "cache": "error"}
        warnings.append(f"Dependency index status unavailable: {exc}")

    try:
        semantic_model = semantic_model_status_for_stats()
    except Exception as exc:
        semantic_model = {
            "modelId": _SEMANTIC_MODEL_ID,
            "provider": "sentence-transformers",
            "available": False,
            "cached": False,
            "requiresDownload": True,
            "message": f"Semantic model status unavailable: {exc}",
        }
        warnings.append(semantic_model["message"])

    try:
        cache = cache_metadata_for_stats()
    except Exception as exc:
        cache = {"schemaVersion": None, "directory": None, "entries": []}
        warnings.append(f"Cache metadata unavailable: {exc}")

    roots: list[str] = []
    for candidate in list(_http_roots) + [
        p_stats.get("base_dir"),
        s_stats.get("base_dir"),
        d_stats.get("base_dir"),
    ]:
        if candidate and candidate not in roots:
            roots.append(candidate)

    indexed_roots = []
    for root in roots:
        try:
            root_path = Path(root).resolve()
            exists = root_path.exists()
            resolved = str(root_path)
        except Exception:
            exists = False
            resolved = str(root)
        indexed_roots.append({
            "path": resolved,
            "exists": exists,
            "fileCount": None,
            "indexedFileCount": p_stats.get("files") if p_stats.get("base_dir") == root else None,
            "baseDir": root,
        })

    pattern_ready = int(p_stats.get("files") or 0) > 0
    semantic_ready = int(s_stats.get("chunks") or 0) > 0
    dependency_ready = int(d_stats.get("files") or 0) > 0
    semantic_available = bool(semantic_model.get("available"))

    if not semantic_available:
        warnings.append("Semantic model is not available locally.")
    if not pattern_ready:
        warnings.append("Pattern index is not ready.")
    if not indexed_roots:
        warnings.append("No indexed roots are configured or loaded.")

    status = "ok" if pattern_ready and indexed_roots else "not_ready"
    if status == "ok" and (not semantic_available or not dependency_ready):
        status = "degraded"

    return {
        "version": VERSION,
        "status": status,
        "device": device,
        "indexedRoots": indexed_roots,
        "indexes": {
            "pattern": {
                "ready": pattern_ready,
                "fileCount": p_stats.get("files"),
                "vramMb": p_stats.get("vram_mb"),
                "cacheStatus": p_stats.get("cache"),
            },
            "semantic": {
                "ready": semantic_ready,
                "chunkCount": s_stats.get("chunks"),
                "vramMb": s_stats.get("vram_mb"),
                "modelId": semantic_model.get("modelId"),
                "modelAvailable": semantic_available,
                "message": semantic_model.get("message"),
            },
            "dependency": {
                "ready": dependency_ready,
                "fileCount": d_stats.get("files"),
                "edgeCount": d_stats.get("edges"),
                "analysisMode": "heuristic",
                "confidence": "medium" if dependency_ready else "low",
            },
        },
        "cache": cache,
        "capabilities": {
            "patternSearch": pattern_ready,
            "semanticSearch": semantic_ready,
            "hybridSearch": pattern_ready or semantic_ready,
            "dependencyImpact": dependency_ready,
            "signalScan": pattern_ready,
            "mcpTools": True,
        },
        "warnings": list(dict.fromkeys(warnings)),
        "limitations": _GLOBAL_LIMITATIONS,
    }


def _make_deps():
    from gpu_dep_index import DepIndex
    return DepIndex()


def _make_symbols():
    from symbol_index import SymbolIndex
    return SymbolIndex()


def _make_planner():
    from change_planner import ChangePlanner
    return ChangePlanner(index, semantic, deps, symbols, git_state)


# ---------------------------------------------------------------------------
# Global shared state
# ---------------------------------------------------------------------------

mcp = FastMCP("gpu-search")
index = _LazyService(_make_index)
semantic = _LazyService(_make_semantic)
deps = _LazyService(_make_deps)
symbols = _LazyService(_make_symbols)
git_state = GitState()
planner = _LazyService(_make_planner)

# Background indexing status — updated by worker threads
_bg_status: dict[str, str] = {
    "pattern": "", "deps": "", "semantic": "", "symbols": "",
}

# Roots we've already loaded semantic cache for (avoid re-loading on every call)
_loaded_roots: set[str] = set()
_http_roots: list[str] = []

# Bounded executor for watchdog-triggered semantic updates
_semantic_update_executor = ThreadPoolExecutor(max_workers=4)

# Result of the most recent POST /index/root call (None if none yet)
_last_index_result: dict | None = None


def _index_root(directory: str, rebuild_cache: bool = False, include_semantic: bool = False) -> dict:
    """Synchronously index a directory for pattern and dependency search.

    Replaces the active root (single-root default). If include_semantic is True,
    attempts to load a pre-built semantic cache — never downloads the model.
    """
    global _http_roots, _bg_status, _last_index_result

    directory = os.path.abspath(directory)
    _http_roots = [directory]

    # --- Pattern index ---
    pattern_ok = False
    pattern_files = 0
    pattern_from_cache = False
    try:
        _bg_status["pattern"] = f"indexing {directory}..."
        p_stats = index.index_directory(
            directory,
            append=False,
            allow_env_files=_ALLOW_ENV_FILES,
            force_rebuild=rebuild_cache,
        )
        pattern_files = int(p_stats.get("indexed", 0))
        # Also check in-memory stats in case cache reported 'indexed' differently
        if pattern_files == 0:
            pattern_files = int(index.stats().get("files", 0))
        pattern_from_cache = p_stats.get("cache") == "loaded"
        pattern_ok = pattern_files > 0
        _bg_status["pattern"] = f"done: {pattern_files} files"
    except Exception as exc:
        _bg_status["pattern"] = f"ERROR: {exc}"

    # --- Dependency index ---
    dep_ok = False
    dep_files = 0
    try:
        _bg_status["deps"] = f"indexing {directory}..."
        d_stats = deps.index_directory(directory, append=False, force_rebuild=rebuild_cache)
        dep_files = int(d_stats.get("files", 0))
        dep_ok = dep_files > 0
        _bg_status["deps"] = f"done: {dep_files} files"
    except Exception as exc:
        _bg_status["deps"] = f"ERROR: {exc}"

    # --- Symbol index ---
    symbol_ok = False
    symbol_count = 0
    symbol_edges = 0
    try:
        _bg_status["symbols"] = f"indexing {directory}..."
        symbol_stats = symbols.index_directory(directory, append=False)
        symbol_count = int(symbol_stats.get("symbols", 0))
        symbol_edges = int(symbol_stats.get("edges", 0))
        symbol_ok = symbol_count > 0
        _bg_status["symbols"] = f"done: {symbol_count} symbols, {symbol_edges} edges"
    except Exception as exc:
        _bg_status["symbols"] = f"ERROR: {exc}"

    # --- Semantic cache (load only, no download) ---
    semantic_ready = False
    semantic_message = "Not requested"
    if include_semantic:
        try:
            s = semantic.try_load_cache(directory)
            if s:
                semantic_ready = True
                semantic_message = f"Loaded {s['chunks']} chunks from cache"
                _bg_status["semantic"] = f"done: {s['chunks']} chunks"
            else:
                semantic_message = "No pre-built cache found; run gpu_semantic_index to build one"
        except Exception as exc:
            semantic_message = f"Cache load failed: {exc}"

    # --- Git state ---
    try:
        git_state.add_root(directory)
    except Exception:
        pass

    ok = pattern_ok
    message = (
        f"Indexed {pattern_files} files from {directory}"
        if ok
        else f"Indexing failed or no files found in {directory}"
    )

    result = {
        "ok": ok,
        "directory": directory,
        "normalizedDirectory": directory,
        "started": True,
        "completed": True,
        "pattern": {
            "ready": pattern_ok,
            "files": pattern_files,
            "fromCache": pattern_from_cache,
        },
        "dependency": {
            "ready": dep_ok,
            "files": dep_files,
        },
        "symbols": {
            "ready": symbol_ok,
            "symbols": symbol_count,
            "edges": symbol_edges,
        },
        "semantic": {
            "requested": include_semantic,
            "ready": semantic_ready,
            "message": semantic_message,
        },
        "message": message,
    }
    _last_index_result = result
    return result


# ---------------------------------------------------------------------------
# File-watcher / debouncer
# ---------------------------------------------------------------------------

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
            if len(path) >= 2 and path[1] == "/":
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
                    print(f"[gpu-search] Auto-loaded semantic cache for {p}: {s['chunks']} chunks",
                          file=sys.stderr, flush=True)

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
        if ext == ".cs":
            _debouncer.submit(f"symbols:{event.src_path}", symbols.update_file, event.src_path)

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
        if Path(event.src_path).suffix.lower() == ".cs":
            _debouncer.submit(f"symbols:{event.src_path}", symbols.update_file, event.src_path)


# ---------------------------------------------------------------------------
# Search helpers (also used by http_server via _app.*)
# ---------------------------------------------------------------------------

def _is_natural_language(query: str) -> bool:
    """Heuristic: multi-word prose queries go to semantic; identifiers/symbols go to pattern."""
    words = query.strip().split()
    if len(words) <= 1:
        return False
    return all(w.isalpha() for w in words)


_SEARCH_MODES = {"auto", "exact", "pattern", "semantic", "hybrid", "symbol", "path"}
_SEARCH_INTENTS = {"locate", "understand", "modify", "debug", "audit"}


def _resolve_search_request(
    query: str,
    mode: str = "auto",
    intent: str = "understand",
    semantic_ready: bool = False,
    symbol_ready: bool = False,
) -> tuple[str, str, list[str]]:
    """Validate a unified search request and return effective mode, intent, warnings."""
    requested_mode = (mode or "auto").strip().lower()
    normalized_intent = (intent or "understand").strip().lower()
    if requested_mode not in _SEARCH_MODES:
        choices = ", ".join(sorted(_SEARCH_MODES))
        raise ValueError(f"Unsupported search mode '{mode}'. Expected one of: {choices}.")
    if normalized_intent not in _SEARCH_INTENTS:
        choices = ", ".join(sorted(_SEARCH_INTENTS))
        raise ValueError(f"Unsupported search intent '{intent}'. Expected one of: {choices}.")

    warnings: list[str] = []
    if requested_mode == "exact":
        effective = "pattern"
    elif requested_mode == "symbol":
        if symbol_ready:
            effective = "symbol"
        else:
            effective = "pattern"
            warnings.append("Symbol index is not ready; using exact pattern search.")
    elif requested_mode == "path":
        effective = "pattern"
        warnings.append("Dedicated path search is not available yet; using exact pattern search.")
    elif requested_mode == "auto":
        wants_semantic = _is_natural_language(query) or normalized_intent in {"modify", "debug"}
        if normalized_intent in {"modify", "debug"} and semantic_ready:
            effective = "hybrid"
        elif wants_semantic and semantic_ready:
            effective = "semantic"
        else:
            effective = "pattern"
            if wants_semantic and not semantic_ready:
                warnings.append("Semantic index is not ready; using exact pattern search.")
    else:
        effective = requested_mode

    if effective == "semantic" and not semantic_ready:
        warnings.append("Semantic index is not ready.")
    elif effective == "hybrid" and not semantic_ready:
        warnings.append("Semantic index is not ready; hybrid search will use exact results only.")

    return effective, normalized_intent, warnings


def _is_test_path(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/").lower()
    name = os.path.basename(normalized)
    return (
        "/tests/" in f"/{normalized.strip('/')}/"
        or "/test/" in f"/{normalized.strip('/')}/"
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name.endswith(".test.js")
        or name.endswith(".test.ts")
        or name.endswith(".test.tsx")
        or name.endswith(".spec.js")
        or name.endswith(".spec.ts")
        or name.endswith(".spec.tsx")
        or name.endswith("tests.cs")
        or name.endswith("test.cs")
    )


def _is_configuration_path(filepath: str) -> bool:
    normalized = filepath.replace("\\", "/").lower()
    name = os.path.basename(normalized)
    return (
        name in {
            "pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini", "package.json",
            "tsconfig.json", "appsettings.json", "appsettings.development.json",
            "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        }
        or name.startswith(".env")
        or name.endswith((".config", ".cfg", ".ini", ".yaml", ".yml", ".toml"))
        or "/config/" in f"/{normalized.strip('/')}/"
        or "/configuration/" in f"/{normalized.strip('/')}/"
    )


def _related_entry(filepath: str, base: str, relation: str, reason: str, **extra) -> dict:
    return {
        "file": _relpath_for_api(filepath, base),
        "absoluteFile": filepath,
        "relation": relation,
        "reason": reason,
        **extra,
    }


def _build_related_files(
    primary_results: list[dict],
    all_results: list[dict],
    base: str,
    include_dependencies: bool,
    include_tests: bool,
) -> tuple[dict[str, list[dict]], list[str]]:
    related: dict[str, list[dict]] = {
        "callers": [],
        "dependencies": [],
        "implementations": [],
        "tests": [],
        "configuration": [],
    }
    warnings: list[str] = []
    seen: dict[str, set[str]] = {key: set() for key in related}

    def add(kind: str, item: dict) -> None:
        key = os.path.normcase(os.path.abspath(item["absoluteFile"]))
        if key not in seen[kind]:
            seen[kind].add(key)
            related[kind].append(item)

    for item in all_results:
        filepath = item.get("absoluteFile")
        if not filepath:
            continue
        if _is_configuration_path(filepath):
            add("configuration", _related_entry(
                filepath, base, "configuration", "configuration-like path matched the query"
            ))
        if include_tests and _is_test_path(filepath):
            add("tests", _related_entry(
                filepath, base, "tested_by", "test-like path matched the query"
            ))

    if include_dependencies:
        dep_stats = deps.stats()
        if dep_stats.get("files", 0) == 0:
            warnings.append("Dependency graph is not ready; callers and dependencies were not expanded.")
        else:
            primary_files = list(dict.fromkeys(
                item["absoluteFile"] for item in primary_results if item.get("absoluteFile")
            ))
            for filepath in primary_files[:5]:
                for dependency in deps.direct_imports(filepath):
                    add("dependencies", _related_entry(
                        dependency, base, "imports", "direct import of a primary result"
                    ))
                for impacted in deps.impact(filepath):
                    if impacted.get("hops") != 1:
                        continue
                    add("callers", _related_entry(
                        impacted["file"],
                        base,
                        "imported_by",
                        impacted.get("reason") or "direct importer of a primary result",
                        hops=1,
                    ))

    if include_tests and not related["tests"]:
        warnings.append(
            "No related tests were identified in the current retrieval results; "
            "symbol-aware test discovery is not available yet."
        )
    return related, warnings

_MAX_FILES = 8
_MAX_MATCHES_PER_FILE = 2
_SNIPPET_CHARS = 300
_MAX_EXPAND_LINES = 60


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

    for r in pattern_results:
        r["_boost"] = git_state.boost(r["file"])
        r["reason"] = "exact token match" + (" + recent git activity" if r["_boost"] else "")
    for r in semantic_results:
        boost = git_state.boost(r["file"])
        r["score"] = round(r["score"] + boost, 4)
        r["reason"] = "semantic match" + (" + recent git activity" if boost else "")

    pattern_results.sort(key=lambda r: r["_boost"], reverse=True)
    semantic_results.sort(key=lambda r: r["score"], reverse=True)

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


def _http_search_structured(
    query: str,
    top_k: int = 5,
    mode: str = "auto",
    context_mode: str = "compact",
    intent: str = "understand",
    include_dependencies: bool = False,
    include_tests: bool = False,
) -> dict:
    pattern_stats = index.stats()
    semantic_stats = semantic.stats()
    symbol_stats = symbols.stats()
    pattern_ready = pattern_stats["files"] > 0
    semantic_ready = semantic_stats["chunks"] > 0
    symbol_ready = symbol_stats["symbols"] > 0
    effective, normalized_intent, warnings = _resolve_search_request(
        query,
        mode=mode,
        intent=intent,
        semantic_ready=semantic_ready,
        symbol_ready=symbol_ready,
    )

    pattern_results: list = []
    semantic_results: list = []
    if effective in {"pattern", "hybrid"} and pattern_ready:
        pattern_results = index.search(query)
        for result in pattern_results:
            boost = git_state.boost(result["file"])
            result["reason"] = "exact token match" + (
                " + recent git activity" if boost else ""
            )
        pattern_results.sort(key=lambda result: git_state.boost(result["file"]), reverse=True)
    elif effective in {"pattern", "hybrid"} and not pattern_ready:
        warnings.append("Pattern index is not ready.")

    if effective in {"semantic", "hybrid"} and semantic_ready:
        semantic_results = semantic.search(query, top_k=top_k)
        for result in semantic_results:
            boost = git_state.boost(result["file"])
            result["score"] = round(result["score"] + boost, 4)
            result["reason"] = "semantic match" + (
                " + recent git activity" if boost else ""
            )
        semantic_results.sort(key=lambda result: result["score"], reverse=True)

    base = (
        pattern_stats.get("base_dir")
        or semantic_stats.get("base_dir")
        or symbol_stats.get("base_dir")
        or ""
    )
    if effective == "symbol":
        results = [
            {
                "file": _relpath_for_api(symbol.file_path, base),
                "absoluteFile": symbol.file_path,
                "lineStart": symbol.line_start,
                "lineEnd": symbol.line_end,
                "score": 1.0,
                "reason": f"{symbol.kind} declaration",
                "snippet": redact(symbol.signature)[:300],
                "engine": "symbol",
                "symbolId": symbol.id,
                "symbolKind": symbol.kind,
                "qualifiedName": symbol.qualified_name,
            }
            for symbol in symbols.find_symbol(query, top_k=top_k)
        ]
    else:
        pattern_files = {result["file"] for result in pattern_results}
        results = _pattern_structured(pattern_results, {"base_dir": base}, context_mode)
        results.extend(_semantic_structured(
            [
                result for result in semantic_results
                if effective != "hybrid" or result["file"] not in pattern_files
            ],
            {"base_dir": base},
            context_mode,
        ))

    include_dependencies = include_dependencies or normalized_intent in {"modify", "debug"}
    include_tests = include_tests or normalized_intent in {"modify", "debug"}
    primary_results = [
        result for result in results
        if not _is_test_path(result.get("absoluteFile", ""))
        and not _is_configuration_path(result.get("absoluteFile", ""))
    ]
    if not primary_results:
        primary_results = list(results)

    related_files, expansion_warnings = _build_related_files(
        primary_results,
        results,
        base,
        include_dependencies=include_dependencies,
        include_tests=include_tests,
    )
    warnings.extend(expansion_warnings)
    warnings = list(dict.fromkeys(warnings))

    return {
        "query": query,
        "mode": effective,
        "contextMode": context_mode,
        "results": results,
        "mode_used": effective,
        "intent": normalized_intent,
        "primary_results": primary_results,
        "related_files": related_files,
        "warnings": warnings,
        "index_status": {
            "pattern_ready": pattern_ready,
            "semantic_ready": semantic_ready,
            "symbol_ready": symbols.stats()["symbols"] > 0,
        },
    }

# ---------------------------------------------------------------------------
# Register MCP tools and prompts
# ---------------------------------------------------------------------------

import mcp_tools as _mcp_tools
_tool_fns = _mcp_tools.register(mcp)

# Re-export tool functions so existing code and tests can reference them
# as mcp_server.<tool_name>.
search_code = _tool_fns["search_code"]
gpu_search = _tool_fns["gpu_search"]
gpu_index = _tool_fns["gpu_index"]
gpu_stats = _tool_fns["gpu_stats"]
gpu_update_file = _tool_fns["gpu_update_file"]
gpu_read_block = _tool_fns["gpu_read_block"]
gpu_skeleton = _tool_fns["gpu_skeleton"]
gpu_semantic_index = _tool_fns["gpu_semantic_index"]
gpu_add_directory = _tool_fns["gpu_add_directory"]
dep_impact = _tool_fns["dep_impact"]
dep_imports = _tool_fns["dep_imports"]
dep_index = _tool_fns["dep_index"]
scan_repository_signals = _tool_fns["scan_repository_signals"]
gpu_semantic_search = _tool_fns["gpu_semantic_search"]
find_symbol = _tool_fns["find_symbol"]
find_implementations = _tool_fns["find_implementations"]
find_references = _tool_fns["find_references"]
find_callers = _tool_fns["find_callers"]
find_callees = _tool_fns["find_callees"]
find_tests = _tool_fns["find_tests"]
explain_impact = _tool_fns["explain_impact"]
plan_change = _tool_fns["plan_change"]

# ---------------------------------------------------------------------------
# Import HTTP handler — must come after all global state and tools are defined
# so that http_server's circular import of mcp_server receives a complete module.
# Re-export for test and external compatibility.
# ---------------------------------------------------------------------------

from http_server import (  # noqa: E402,F401 (re-exported for mcp_server.* compatibility)
    _HttpApi,
    _require_under_root,
    _active_roots,
    _run_signal,
)


# ---------------------------------------------------------------------------
# CLI and startup
# ---------------------------------------------------------------------------

def _client_configuration_status() -> dict:
    """Return configuration presence only; never read or expose client config contents."""
    home = Path.home()
    candidates = {
        "claude": [home / ".claude.json"],
        "codex": [home / ".codex" / "config.toml", home / ".codex" / "config.yaml"],
    }
    return {
        client: {
            "configured": any(path.exists() for path in paths),
            "paths": [str(path) for path in paths],
        }
        for client, paths in candidates.items()
    }


def _probe_local_http(port: int = 8765) -> dict:
    """Probe the loopback health endpoint without starting or modifying the server."""
    connection = HTTPConnection("127.0.0.1", port, timeout=0.25)
    try:
        connection.request("GET", "/health")
        response = connection.getresponse()
        response.read()
        return {
            "url": f"http://127.0.0.1:{port}",
            "reachable": True,
            "statusCode": response.status,
        }
    except (OSError, HTTPException) as exc:
        return {
            "url": f"http://127.0.0.1:{port}",
            "reachable": False,
            "statusCode": None,
            "message": str(exc),
        }
    finally:
        connection.close()


def doctor_snapshot(port: int = 8765) -> dict:
    """Build a read-only diagnostic report for the CLI."""
    report = diagnostics_snapshot()
    report["system"] = {
        "os": platform.system(),
        "osVersion": platform.release(),
        "machine": platform.machine(),
        "pythonVersion": platform.python_version(),
        "pythonExecutable": sys.executable,
    }
    try:
        report["configuredRoots"] = [
            {"path": str(Path(root).resolve()), "exists": Path(root).exists()}
            for root in _load_config_dirs()
        ]
    except Exception as exc:
        report["configuredRoots"] = []
        report.setdefault("warnings", []).append(
            f"Configured root status unavailable: {exc}"
        )
    report["clientConfiguration"] = _client_configuration_status()
    report["http"] = _probe_local_http(port)
    return report


def _format_doctor_report(report: dict) -> str:
    """Format safe diagnostic metadata for humans without source or secret values."""
    system = report.get("system", {})
    device = report.get("device", {})
    indexes = report.get("indexes", {})
    clients = report.get("clientConfiguration", {})
    http = report.get("http", {})
    lines = [
        f"gpu-search-mcp doctor — {report.get('status', 'unknown')}",
        f"Version: {report.get('version', 'unknown')}",
        (
            "System: "
            f"{system.get('os', 'unknown')} {system.get('osVersion', '')} "
            f"({system.get('machine', 'unknown')}), Python "
            f"{system.get('pythonVersion', 'unknown')}"
        ).strip(),
        (
            "Device: "
            f"{device.get('backend', 'unknown')} "
            f"({device.get('torchDevice', 'unknown')})"
        ),
    ]
    for name in ("pattern", "semantic", "dependency"):
        state = indexes.get(name, {})
        lines.append(
            f"{name.title()} index: "
            f"{'ready' if state.get('ready') else 'not ready'}"
        )

    roots = report.get("indexedRoots", [])
    lines.append(f"Indexed roots: {len(roots)}")
    for root in roots:
        lines.append(f"  - {root.get('path')}")

    configured_roots = report.get("configuredRoots", [])
    lines.append(f"Configured roots: {len(configured_roots)}")
    for root in configured_roots:
        suffix = "" if root.get("exists") else " (missing)"
        lines.append(f"  - {root.get('path')}{suffix}")

    lines.append("Client configuration:")
    for client, state in clients.items():
        lines.append(
            f"  - {client}: {'found' if state.get('configured') else 'not found'}"
        )

    lines.append(
        "HTTP: "
        f"{'reachable' if http.get('reachable') else 'not reachable'} "
        f"({http.get('url', 'http://127.0.0.1:8765')})"
    )
    warnings = report.get("warnings", [])
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in warnings)
    return "\n".join(lines)


def _run_doctor(args) -> int:
    report = doctor_snapshot(port=args.port)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_doctor_report(report))
    return 0 if report.get("status") in {"ok", "degraded"} else 1

def _parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "setup":
        parser = argparse.ArgumentParser(
            prog="gpu-search-mcp setup",
            description="Safely configure gpu-search-mcp for supported local clients.",
        )
        parser.add_argument(
            "--client", action="append", dest="clients", choices=("claude", "codex"),
            help="Client to configure (repeatable). Defaults to detected clients.",
        )
        parser.add_argument(
            "--directory", "-d", action="append", dest="directories", metavar="DIR",
            help="Directory to index (repeatable; defaults to the current directory).",
        )
        parser.add_argument("--no-index", action="store_true", help="Do not add an index directory.")
        parser.add_argument(
            "--no-model", action="store_true", help="Skip the local-only model cache check."
        )
        parser.add_argument(
            "--dry-run", action="store_true", help="Show the write plan without changing files."
        )
        parser.add_argument(
            "--yes", "-y", action="store_true", help="Apply without an interactive confirmation."
        )
        args = parser.parse_args(argv[1:])
        args.command = "setup"
        return args

    if argv and argv[0] == "doctor":
        parser = argparse.ArgumentParser(
            prog="gpu-search-mcp doctor",
            description="Read-only local diagnostics; does not index, download, or modify files.",
        )
        parser.add_argument(
            "--json",
            action="store_true",
            dest="json_output",
            help="Emit the diagnostic report as JSON.",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=8765,
            help="Loopback HTTP port to probe (default: 8765).",
        )
        args = parser.parse_args(argv[1:])
        args.command = "doctor"
        return args

    parser = argparse.ArgumentParser(
        description="gpu-search-mcp: GPU-accelerated code search MCP server"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )
    parser.add_argument("--directory", "-d", action="append", dest="directories",
                        metavar="DIR", help="Directory to index (repeat for multi-root)")
    parser.add_argument(
        "--allow-env-files", action="store_true", default=False,
        help="Opt in to indexing .env files (excluded by default for security)",
    )
    parser.add_argument("--http", action="store_true", help="Run HTTP API instead of MCP stdio")
    parser.add_argument(
        "--rebuild-cache", action="store_true",
        help="Ignore existing persistent caches and write fresh cache metadata.",
    )
    parser.add_argument(
        "--semantic-model", default=None, metavar="MODEL_ID",
        help="Sentence-transformers embedding model for semantic search.",
    )
    parser.add_argument(
        "--download-semantic-model", action="store_true",
        help="Explicitly download/preload the configured semantic embedding model.",
    )
    parser.add_argument(
        "--force-download-semantic-model", action="store_true",
        help="Force refresh/preload behavior when used with --download-semantic-model.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="HTTP bind host. Defaults to 127.0.0.1; use 0.0.0.0 only explicitly.")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port")
    parser.add_argument(
        "--device", default=None, choices=["auto", "cuda", "mps", "cpu"],
        metavar="DEVICE",
        help="Torch compute backend: auto (default) | cuda | mps | cpu. "
             "Auto selects cuda > mps > cpu. Also reads GPU_SEARCH_DEVICE env var.",
    )
    args = parser.parse_args(argv)
    args.command = "serve"
    return args

def _prepare_startup(args):
    global _ALLOW_ENV_FILES, _REBUILD_CACHE, _SEMANTIC_MODEL_ID
    _REBUILD_CACHE = bool(getattr(args, "rebuild_cache", False))
    _SEMANTIC_MODEL_ID = resolve_semantic_model_id(getattr(args, "semantic_model", None))
    set_configured_semantic_model_id(_SEMANTIC_MODEL_ID)
    device_pref = getattr(args, "device", None) or os.environ.get("GPU_SEARCH_DEVICE") or "auto"
    os.environ["GPU_SEARCH_DEVICE"] = device_pref
    if args.allow_env_files:
        _ALLOW_ENV_FILES = True
        INDEXED_EXTS.add('.env')
        print("[gpu-search] WARNING: .env indexing enabled — search results may contain secrets",
              file=sys.stderr, flush=True)

    if getattr(args, "download_semantic_model", False) and not args.directories:
        print("[gpu-search] Semantic model download requested without indexing directories",
              file=sys.stderr, flush=True)
        return [], []

    cli_dirs = [os.path.abspath(d) for d in (args.directories or [])]
    config_dirs = _load_config_dirs()
    extra_dirs = [d for d in config_dirs if d not in cli_dirs and os.path.isdir(d)]

    cli_targets = [t for t in cli_dirs if os.path.isdir(t)]
    # When --directory is explicitly supplied, use only those directories for all
    # indexes — including semantic cache loading.  Merging config-saved roots from
    # previous sessions (extra_dirs) into all_targets causes cross-repo result
    # contamination: semantic chunks from an old repo are loaded into the current
    # session and appear in search/signal-scan results.
    if args.directories:
        all_targets = cli_targets
    else:
        all_targets = cli_targets + extra_dirs

    if cli_dirs:
        _save_config_dirs(cli_dirs)
    print(f"[gpu-search] Index targets: {cli_targets}  |  Semantic cache: {all_targets}",
          file=sys.stderr, flush=True)
    return cli_targets, all_targets


def _start_indexes(cli_targets: list[str], all_targets: list[str]):
    index._get()
    deps._get()
    semantic._get()
    symbols._get()

    observer = _make_observer()
    for target in cli_targets:
        observer.schedule(_Watcher(), target, recursive=True)

    def _startup_pattern():
        for i, target in enumerate(cli_targets):
            try:
                _bg_status["pattern"] = f"indexing {target}..."
                stats = index.index_directory(
                    target, append=(i > 0), allow_env_files=_ALLOW_ENV_FILES,
                    force_rebuild=_REBUILD_CACHE,
                )
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
                dep_stats = deps.index_directory(
                    target, append=(i > 0), force_rebuild=_REBUILD_CACHE
                )
                _bg_status["deps"] = f"done: {dep_stats['files']} files, {dep_stats['edges']} edges"
                print(f"[gpu-search] Dep graph: {dep_stats['files']} files, {dep_stats['edges']} edges from {target}",
                      file=sys.stderr)
            except Exception as e:
                _bg_status["deps"] = f"ERROR: {e}"
                print(f"[gpu-search] Dep index FAILED: {e}", file=sys.stderr, flush=True)

    def _startup_symbols():
        for i, target in enumerate(cli_targets):
            try:
                _bg_status["symbols"] = f"indexing {target}..."
                symbol_stats = symbols.index_directory(target, append=(i > 0))
                _bg_status["symbols"] = (
                    f"done: {symbol_stats['symbols']} symbols, {symbol_stats['edges']} edges"
                )
                print(
                    f"[gpu-search] Symbol index: {symbol_stats['symbols']} symbols, "
                    f"{symbol_stats['edges']} edges from {target}",
                    file=sys.stderr,
                )
            except Exception as e:
                _bg_status["symbols"] = f"ERROR: {e}"
                print(f"[gpu-search] Symbol index FAILED: {e}", file=sys.stderr, flush=True)

    def _startup_semantic():
        for i, target in enumerate(all_targets):
            if _REBUILD_CACHE:
                continue
            s = semantic.try_load_cache(target) if i == 0 else semantic.merge_cache(target)
            if s:
                _loaded_roots.add(os.path.abspath(target))
                print(f"[gpu-search] Semantic cache merged: {s['chunks']} chunks ({s['vram_mb']} MB) total",
                      file=sys.stderr, flush=True)

    for target in cli_targets:
        git_state.add_root(target)

    threading.Thread(target=_startup_pattern, daemon=True).start()
    threading.Thread(target=_startup_deps, daemon=True).start()
    threading.Thread(target=_startup_symbols, daemon=True).start()
    threading.Thread(target=_startup_semantic, daemon=True).start()
    observer.daemon = True
    observer.start()
    return observer


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
    if getattr(args, "download_semantic_model", False):
        requested_device = os.environ.get("GPU_SEARCH_DEVICE") or None
        if requested_device == "auto":
            requested_device = None
        status = download_semantic_model(
            _SEMANTIC_MODEL_ID,
            device=requested_device,
            force=bool(getattr(args, "force_download_semantic_model", False)),
        )
        print(f"[gpu-search] {status.get('message')}", file=sys.stderr, flush=True)
        if not args.directories:
            if not status.get("available"):
                raise SystemExit(1)
            return
    if args.http:
        _start_http(args, cli_targets, all_targets)
        return
    _start_indexes(cli_targets, all_targets)
    mcp.run(transport="stdio")


def cli_main():
    """Entry point for the gpu-search-mcp CLI command."""
    args = _parse_args()
    if args.command == "setup":
        from .setup_cli import run_setup

        return run_setup(args)
    if args.command == "doctor":
        return _run_doctor(args)
    _start_server(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
