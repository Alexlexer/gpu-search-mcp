import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import torch

from gpu_index import SKIP_DIRS, _best_device

from cache_manager import (
    DEPENDENCY_CACHE_SCHEMA_VERSION,
    cache_transaction,
    compute_source_fingerprint,
    invalidate_cache_entry,
    is_cache_entry_valid,
    load_cache_metadata,
    upsert_cache_entry,
)
from server_config import VERSION

DEVICE = _best_device()

# Per-extension import extractors → list of raw module/path strings
_PATTERNS: dict[str, list[re.Pattern]] = {
    ".py": [
        re.compile(r"^\s*from\s+([\w.]+)\s+import", re.MULTILINE),
        re.compile(r"^\s*import\s+([\w.,\s]+)", re.MULTILINE),
    ],
    ".js":  [re.compile(r"""(?:import\s+(?:.*?\s+from\s+)?|require\s*\(\s*)['"]([^'"]+)['"]""")],
    ".ts":  [re.compile(r"""(?:import\s+(?:.*?\s+from\s+)?|require\s*\(\s*)['"]([^'"]+)['"]""")],
    ".tsx": [re.compile(r"""(?:import\s+(?:.*?\s+from\s+)?|require\s*\(\s*)['"]([^'"]+)['"]""")],
    ".jsx": [re.compile(r"""(?:import\s+(?:.*?\s+from\s+)?|require\s*\(\s*)['"]([^'"]+)['"]""")],
    ".go":  [re.compile(r'"([^"]+)"')],
    ".rs":  [re.compile(r"^\s*(?:use|mod)\s+([\w:]+)", re.MULTILINE)],
    ".java":[re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)(?:\.\*)?;", re.MULTILINE)],
    ".cs":  [re.compile(r"^\s*(?:global\s+)?using\s+(?:static\s+)?(?:\w+\s*=\s*)?([\w.]+);", re.MULTILINE)],
    ".rb":  [re.compile(r"""^\s*require(?:_relative)?\s+['"]([^'"]+)['"]""", re.MULTILINE)],
}

_DEP_EXTS = set(_PATTERNS.keys())

# JS/TS extensions to try when resolving bare relative paths
_JS_EXTS = [".ts", ".tsx", ".js", ".jsx"]
_JS_INDEX = ["index.ts", "index.tsx", "index.js", "index.jsx"]
_DEP_CACHE_VERSION = DEPENDENCY_CACHE_SCHEMA_VERSION


def _dependency_cache_components() -> dict:
    return {
        "parser": "regex-imports-v2",
        "edgeFormat": "sparse-coo-v1",
    }


def _load_aliases(directory: str) -> dict[str, list[str]]:
    """
    Read tsconfig.json / jsconfig.json compilerOptions.paths and baseUrl.
    Returns {alias_prefix: [resolved_root, ...]} for non-relative module resolution.
    """
    aliases: dict[str, list[str]] = {}
    for config_name in ("tsconfig.json", "jsconfig.json", "tsconfig.base.json"):
        config_path = Path(directory) / config_name
        if not config_path.exists():
            continue
        try:
            data = json.loads(config_path.read_text(encoding="utf-8", errors="replace"))
            opts = data.get("compilerOptions", {})
            base_url = opts.get("baseUrl", ".")
            base_dir = str((config_path.parent / base_url).resolve())
            paths = opts.get("paths", {})
            for pattern, targets in paths.items():
                # Strip trailing /* from pattern key
                key = pattern.rstrip("/*").rstrip("/")
                resolved = []
                for t in targets:
                    t_root = t.rstrip("/*").rstrip("/")
                    resolved.append(str((config_path.parent / base_url / t_root).resolve()))
                if key and resolved:
                    aliases[key] = resolved
            # baseUrl itself acts as an alias root (bare imports resolve from it)
            if base_dir not in aliases.get("", []):
                aliases.setdefault("", []).append(base_dir)
        except Exception:
            pass
    return aliases


def _extract_raw(fpath: str, text: str) -> list[str]:
    ext = Path(fpath).suffix.lower()
    patterns = _PATTERNS.get(ext, [])
    raw = []
    for pat in patterns:
        for m in pat.finditer(text):
            for g in m.groups():
                if g:
                    for part in g.split(","):
                        raw.append(part.strip())
    return raw


def _raw_import_reason(fpath: str, raw: str) -> str:
    ext = Path(fpath).suffix.lower()
    if ext == ".py":
        return f"imports module {raw}"
    if ext in (".js", ".ts", ".tsx", ".jsx"):
        return f"imports module {raw}"
    if ext == ".cs":
        return f"imports namespace {raw}"
    return "heuristic dependency edge"


def _extract_csharp_symbols(fpath: str, text: str) -> dict[str, list[str]]:
    if Path(fpath).suffix.lower() != ".cs":
        return {}
    symbols: dict[str, list[str]] = {
        "usings": [],
        "namespaces": [],
        "types": [],
        "interfaces": [],
        "base_types": [],
    }
    for m in re.finditer(r"^\s*(?:global\s+)?using\s+(?:static\s+)?(?:\w+\s*=\s*)?([\w.]+);", text, re.MULTILINE):
        symbols["usings"].append(m.group(1))
    for m in re.finditer(r"\bnamespace\s+([\w.]+)", text):
        symbols["namespaces"].append(m.group(1))
    type_pat = re.compile(
        r"\b(?:public|private|protected|internal|sealed|abstract|static|partial|readonly|file|\s)*"
        r"(class|interface|record|struct|enum)\s+(\w+)(?:\s*:\s*([^{;\n]+))?",
        re.MULTILINE,
    )
    for m in type_pat.finditer(text):
        kind, name, bases = m.groups()
        symbols["types"].append(name)
        if kind == "interface":
            symbols["interfaces"].append(name)
        if bases:
            for b in bases.split(","):
                symbols["base_types"].append(b.strip().split(".")[-1].split("<")[0].strip())
    return {k: sorted(set(v)) for k, v in symbols.items() if v}


def _contains_type_reference(text: str, type_name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(type_name)}\b", text))


def _csharp_reason_for_target(src_info: dict[str, list[str]], src_text: str,
                              target_info: dict[str, list[str]], fallback: str) -> str:
    for target_type in target_info.get("types", []):
        if _contains_type_reference(src_text, target_type):
            if target_type in src_info.get("base_types", []):
                if target_type in target_info.get("interfaces", []):
                    return f"implements interface {target_type}"
                return f"inherits from {target_type}"
            return f"references type {target_type}"
    for used_ns in src_info.get("usings", []):
        if used_ns in target_info.get("namespaces", []):
            return f"imports namespace {used_ns}"
    return fallback


def _js_candidates(base: str) -> list[str]:
    """All TS/JS extensions + index files to try for a base path."""
    return (
        [base + e for e in _JS_EXTS]
        + [base.removesuffix(".js") + e for e in _JS_EXTS]  # .js→.ts substitution
        + [os.path.join(base, idx) for idx in _JS_INDEX]
        + [base]
    )


def _resolve(raw: str, src_file: str, file_set: set[str],
             aliases: Optional[dict[str, list[str]]] = None) -> Optional[str]:
    """Try to resolve a raw import string to an absolute path in the project."""
    src_dir = os.path.dirname(src_file)
    ext = Path(src_file).suffix.lower()

    # Relative imports (starts with . or /)
    if raw.startswith(".") or raw.startswith("/"):
        base = os.path.normpath(os.path.join(src_dir, raw))
        candidates = _js_candidates(base) if ext in (".js", ".ts", ".tsx", ".jsx") else [base]
        for c in candidates:
            if c in file_set:
                return c
        return None

    # Python relative (from .foo import bar → raw = ".foo")
    if raw.startswith("."):
        joined = os.path.normpath(os.path.join(src_dir, raw.lstrip(".").replace(".", os.sep) + ".py"))
        return joined if joined in file_set else None

    # TS/JS: try path aliases from tsconfig then baseUrl roots
    if ext in (".ts", ".tsx", ".js", ".jsx") and aliases:
        # Longest matching alias prefix first
        for prefix in sorted(aliases.keys(), key=len, reverse=True):
            if not prefix:
                continue
            if raw == prefix or raw.startswith(prefix + "/"):
                suffix = raw[len(prefix):]
                for root in aliases[prefix]:
                    base = os.path.normpath(root + suffix)
                    for c in _js_candidates(base):
                        if c in file_set:
                            return c
        # baseUrl bare import (empty prefix key)
        for root in aliases.get("", []):
            base = os.path.normpath(os.path.join(root, raw))
            for c in _js_candidates(base):
                if c in file_set:
                    return c

    # Dot-notation module/namespace → try matching project files by path suffix
    suffix_map = {".py": [".py"], ".java": [".java"], ".cs": [".cs"], ".rs": [".rs"]}
    if ext in suffix_map:
        for file_ext in suffix_map[ext]:
            as_path = raw.replace(".", os.sep) + file_ext
            for f in file_set:
                if f.endswith(as_path):
                    return f

    return None


class DepIndex:
    def __init__(self):
        self._files: list[str] = []
        self._file_idx: dict[str, int] = {}
        # Sparse COO tensor: adj[i,j]=1 means file i imports file j.
        # Orders of magnitude smaller than dense for real codebases.
        self._adj: Optional[torch.Tensor] = None
        self._edges: list[tuple[int, int]] = []   # kept for incremental updates
        self._edge_reasons: dict[tuple[int, int], str] = {}
        self._cs_symbols: dict[str, dict[str, list[str]]] = {}
        self._cache_status = "cold"
        self.base_dir: Optional[str] = None
        self._lock = threading.Lock()

    def _cache_dir(self, directory: str) -> Path:
        return Path(directory) / ".gpu-search-cache"

    def _signature(self, fpath: str) -> Optional[dict]:
        try:
            st = os.stat(fpath)
            return {"path": fpath, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            return None

    def _try_load_cache(self, directory: str, files: list[str]) -> Optional[dict]:
        try:
            cache_path = self._cache_dir(directory) / "dep-graph-v1.json"
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("version") != _DEP_CACHE_VERSION or data.get("files") != files:
                return None
            # Reject cache written for a different directory (cross-repo leakage guard).
            cached_dir = data.get("directory")
            if cached_dir and Path(cached_dir).resolve() != Path(directory).resolve():
                return None
            sigs = data.get("signatures", {})
            for f in files:
                sig = self._signature(f)
                cached = sigs.get(f)
                if sig is None or cached is None:
                    return None
                if sig["size"] != cached.get("size") or sig["mtime_ns"] != cached.get("mtime_ns"):
                    return None
            self._cache_status = "loaded"
            return data
        except Exception:
            return None

    def _write_cache(self, directory: str, files: list[str], edges: list[tuple[int, int]],
                     cs_symbols: dict[str, dict[str, list[str]]],
                     edge_reasons: dict[tuple[int, int], str],
                     source_fingerprint: dict | None = None):
        try:
            cache_dir = self._cache_dir(directory)
            sigs = {f: self._signature(f) for f in files}
            data = {
                "version": _DEP_CACHE_VERSION,
                "directory": directory,
                "files": files,
                "signatures": sigs,
                "edges": edges,
                "edge_reasons": {
                    f"{source}:{target}": reason
                    for (source, target), reason in edge_reasons.items()
                },
                "csharp_symbols": cs_symbols,
                "updated_at": time.time(),
            }
            cache_path = cache_dir / "dep-graph-v1.json"
            fingerprint = source_fingerprint or compute_source_fingerprint(
                directory,
                _DEP_EXTS,
                SKIP_DIRS,
                settings={"cache": "dependency"},
            )
            with cache_transaction(cache_dir, "dependency") as transaction:
                transaction.stage_json(cache_path, data)
                upsert_cache_entry(
                    cache_dir,
                    directory,
                    VERSION,
                    name="dependency",
                    schema_version=DEPENDENCY_CACHE_SCHEMA_VERSION,
                    file_path=cache_path,
                    source_fingerprint=fingerprint,
                    status="rebuilt",
                    components=_dependency_cache_components(),
                    transaction=transaction,
                )
        except Exception:
            pass
    def _build_sparse(self, N: int, edges: list[tuple[int, int]]) -> Optional[torch.Tensor]:
        if DEVICE.type == "cpu":
            return None
        if not edges:
            idx = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
            vals = torch.zeros(0, dtype=torch.float32, device=DEVICE)
        else:
            idx = torch.tensor(edges, dtype=torch.long, device=DEVICE).T
            vals = torch.ones(len(edges), dtype=torch.float32, device=DEVICE)
        return torch.sparse_coo_tensor(
            idx.contiguous(), vals, (N, N), device=DEVICE, check_invariants=True
        ).coalesce()

    def index_directory(
        self, directory: str, max_file_mb: float = 5.0, append: bool = False,
        force_rebuild: bool = False,
    ) -> dict:
        directory = os.path.abspath(directory)
        max_bytes = int(max_file_mb * 1024 * 1024)

        new_files: list[str] = []
        for root, dirs, fnames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in fnames:
                if Path(fname).suffix.lower() in _DEP_EXTS:
                    fpath = os.path.join(root, fname)
                    try:
                        if 0 < os.path.getsize(fpath) <= max_bytes:
                            new_files.append(fpath)
                    except Exception:
                        pass

        with self._lock:
            existing_files = list(self._files)
        if append and existing_files:
            existing = [f for f in existing_files if not f.startswith(directory)]
            files = existing + new_files
        else:
            files = new_files

        if not files:
            return {"files": 0, "edges": 0}

        file_set = set(files)
        local_file_idx = {f: i for i, f in enumerate(files)}
        N = len(files)
        source_fingerprint = None
        if not append:
            source_fingerprint = compute_source_fingerprint(
                directory,
                _DEP_EXTS,
                SKIP_DIRS,
                max_file_mb=max_file_mb,
                settings={"cache": "dependency"},
            )
            metadata = load_cache_metadata(self._cache_dir(directory))
            entry_valid = is_cache_entry_valid(
                metadata,
                "dependency",
                DEPENDENCY_CACHE_SCHEMA_VERSION,
                source_fingerprint,
                VERSION,
                _dependency_cache_components(),
            )
            if force_rebuild:
                invalidate_cache_entry(self._cache_dir(directory), "dependency", "rebuild_requested")
            elif metadata is not None and not entry_valid:
                invalidate_cache_entry(self._cache_dir(directory), "dependency", "stale")
            cached = None
            if not force_rebuild and entry_valid:
                cached = self._try_load_cache(directory, files)
            if cached is not None:
                edges = [tuple(e) for e in cached.get("edges", [])]
                edge_reasons = {}
                for key, reason in cached.get("edge_reasons", {}).items():
                    try:
                        s, t = key.split(":", 1)
                        edge_reasons[(int(s), int(t))] = reason
                    except Exception:
                        pass
                adj = self._build_sparse(N, edges)
                with self._lock:
                    self._files = files
                    self._file_idx = local_file_idx
                    self._edges = edges
                    self._edge_reasons = edge_reasons
                    self._adj = adj
                    self._cs_symbols = cached.get("csharp_symbols", {})
                    self.base_dir = directory
                return {"files": N, "edges": len(edges), "cache": "loaded"}

        aliases = _load_aliases(directory)
        if aliases:
            n_alias = sum(len(v) for v in aliases.values())
            print(f"[deps] Loaded {n_alias} alias roots from tsconfig", file=sys.stderr, flush=True)

        print(f"[deps] Parsing {N} files...", file=sys.stderr, flush=True)

        edges: list[tuple[int, int]] = []
        edge_reasons: dict[tuple[int, int], str] = {}
        cs_symbols: dict[str, dict[str, list[str]]] = {}
        text_by_file: dict[str, str] = {}
        for i, fpath in enumerate(files):
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                text_by_file[fpath] = text
                cs_info = _extract_csharp_symbols(fpath, text)
                if cs_info:
                    cs_symbols[fpath] = cs_info
                for raw in _extract_raw(fpath, text):
                    resolved = _resolve(raw, fpath, file_set, aliases)
                    if resolved and resolved in local_file_idx and resolved != fpath:
                        edge = (i, local_file_idx[resolved])
                        edges.append(edge)
                        edge_reasons.setdefault(edge, _raw_import_reason(fpath, raw))
            except Exception:
                pass

        # C#: supplement using/namespace imports with namespace/type/base-type mapping.
        ns_to_files: dict[str, set[str]] = {}
        type_to_files: dict[str, set[str]] = {}
        for f, info in cs_symbols.items():
            for ns in info.get("namespaces", []):
                ns_to_files.setdefault(ns, set()).add(f)
            for typ in info.get("types", []):
                type_to_files.setdefault(typ, set()).add(f)
        seen_edges = set(edges)
        for src, info in cs_symbols.items():
            i = local_file_idx[src]
            src_text = text_by_file.get(src, "")
            for used_ns in info.get("usings", []):
                for dst in ns_to_files.get(used_ns, set()):
                    if dst != src:
                        edge = (i, local_file_idx[dst])
                        seen_edges.add(edge)
                        reason = _csharp_reason_for_target(
                            info,
                            src_text,
                            cs_symbols.get(dst, {}),
                            f"imports namespace {used_ns}",
                        )
                        edge_reasons[edge] = reason
            for base_type in info.get("base_types", []):
                for dst in type_to_files.get(base_type, set()):
                    if dst != src:
                        edge = (i, local_file_idx[dst])
                        seen_edges.add(edge)
                        target_info = cs_symbols.get(dst, {})
                        if base_type in target_info.get("interfaces", []):
                            reason = f"implements interface {base_type}"
                        else:
                            reason = f"inherits from {base_type}"
                        edge_reasons[edge] = reason
        edges = sorted(seen_edges)

        adj = self._build_sparse(N, edges)
        vram_kb = round((len(edges) * 3 * 4) / 1024, 1)
        print(f"[deps] {N} files, {len(edges)} edges, ~{vram_kb} KB graph storage", file=sys.stderr, flush=True)
        with self._lock:
            self._files = files
            self._file_idx = local_file_idx
            self._edges = edges
            self._edge_reasons = edge_reasons
            self._adj = adj
            self._cs_symbols = cs_symbols
            self._cache_status = "rebuilt"
            if not append:
                self.base_dir = directory
        if not append:
            self._write_cache(directory, files, edges, cs_symbols, edge_reasons, source_fingerprint)
        return {"files": N, "edges": len(edges), "cache": self._cache_status}

    def update_file(self, fpath: str):
        """Re-parse one file's imports and rebuild the sparse matrix."""
        fpath = os.path.abspath(fpath)
        with self._lock:
            if fpath not in self._file_idx:
                return
            i = self._file_idx[fpath]
            edges = [(s, t) for s, t in self._edges if s != i]
            edge_reasons = {edge: reason for edge, reason in self._edge_reasons.items() if edge[0] != i}
            file_set = set(self._files)
            base_dir = self.base_dir
            n_files = len(self._files)
            local_file_idx = dict(self._file_idx)
        try:
            text = Path(fpath).read_text(encoding="utf-8", errors="replace")
            aliases = _load_aliases(base_dir) if base_dir else {}
            for raw in _extract_raw(fpath, text):
                resolved = _resolve(raw, fpath, file_set, aliases)
                if resolved and resolved in local_file_idx and resolved != fpath:
                    edge = (i, local_file_idx[resolved])
                    edges.append(edge)
                    edge_reasons.setdefault(edge, _raw_import_reason(fpath, raw))
        except Exception:
            pass
        adj = self._build_sparse(n_files, edges)
        with self._lock:
            self._edges = edges
            self._edge_reasons = edge_reasons
            self._adj = adj

    def impact(self, fpath: str, max_hops: int = 20) -> list[dict]:
        """Find transitive importers using CPU edge traversal or GPU sparse BFS."""
        fpath = os.path.abspath(fpath)
        with self._lock:
            if fpath not in self._file_idx:
                return []
            target = self._file_idx[fpath]
            N = len(self._files)
            adj_snapshot = self._adj
            edges_snapshot = list(self._edges)
            files_snapshot = list(self._files)
            edge_reasons_snapshot = dict(self._edge_reasons)

        if DEVICE.type == "cpu":
            rows = [source for source, _ in edges_snapshot]
            columns = [imported for _, imported in edges_snapshot]
            importers: dict[int, list[int]] = {}
            for source, imported in zip(rows, columns):
                importers.setdefault(imported, []).append(source)

            reached = {target}
            frontier = set(importers.get(target, []))
            results: list[dict] = []
            for hop in range(1, max_hops + 1):
                new = sorted(frontier - reached)
                if not new:
                    break
                reached.update(new)
                for idx in new:
                    reason = edge_reasons_snapshot.get((idx, target)) if hop == 1 else None
                    if reason is None:
                        reason = (
                            "reverse dependency edge"
                            if hop == 1
                            else "reachable through dependency graph"
                        )
                    results.append({"file": files_snapshot[idx], "hops": hop, "reason": reason})
                frontier = {
                    source
                    for imported in new
                    for source in importers.get(imported, [])
                }
            return results

        if adj_snapshot is None:
            return []
        indices = adj_snapshot.indices()
        col_mask = indices[1] == target
        frontier = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        frontier[indices[0][col_mask]] = 1.0
        frontier[target] = 0.0

        reached = torch.zeros(N, dtype=torch.float32, device=DEVICE)
        results: list[dict] = []
        for hop in range(1, max_hops + 1):
            new = frontier * (1 - reached)
            if new.sum() == 0:
                break
            reached += new
            for idx in new.nonzero(as_tuple=True)[0].cpu().tolist():
                reason = edge_reasons_snapshot.get((idx, target)) if hop == 1 else None
                if reason is None:
                    reason = "reverse dependency edge" if hop == 1 else "reachable through dependency graph"
                results.append({"file": files_snapshot[idx], "hops": hop, "reason": reason})
            frontier = (torch.sparse.mm(adj_snapshot, new.unsqueeze(1)).squeeze(1) > 0).float()
            frontier[target] = 0.0

        return results

    def direct_imports(self, fpath: str) -> list[str]:
        """Return files that fpath directly imports."""
        fpath = os.path.abspath(fpath)
        with self._lock:
            if fpath not in self._file_idx:
                return []
            i = self._file_idx[fpath]
            if DEVICE.type == "cpu":
                return [self._files[target] for source, target in self._edges if source == i]
            if self._adj is None:
                return []
            indices = self._adj.indices()
            row_mask = indices[0] == i
            return [self._files[j] for j in indices[1][row_mask].cpu().tolist()]

    def csharp_symbols(self, fpath: str) -> dict[str, list[str]]:
        fpath = os.path.abspath(fpath)
        with self._lock:
            return dict(self._cs_symbols.get(fpath, {}))

    def stats(self) -> dict:
        with self._lock:
            N = len(self._files)
            edges = len(self._edges)
            base_dir = self.base_dir
            cache = self._cache_status
        return {"files": N, "edges": edges, "base_dir": base_dir, "cache": cache}
