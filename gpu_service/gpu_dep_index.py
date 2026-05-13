import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional
import torch

from gpu_index import SKIP_DIRS, _best_device

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
        self.base_dir: Optional[str] = None
        self._lock = threading.Lock()

    def _build_sparse(self, N: int, edges: list[tuple[int, int]]) -> torch.Tensor:
        if not edges:
            idx = torch.zeros((2, 0), dtype=torch.long, device=DEVICE)
            vals = torch.zeros(0, dtype=torch.float32, device=DEVICE)
        else:
            idx = torch.tensor(edges, dtype=torch.long, device=DEVICE).T
            vals = torch.ones(len(edges), dtype=torch.float32, device=DEVICE)
        return torch.sparse_coo_tensor(idx, vals, (N, N), device=DEVICE).coalesce()

    def index_directory(self, directory: str, max_file_mb: float = 5.0, append: bool = False) -> dict:
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

        aliases = _load_aliases(directory)
        if aliases:
            n_alias = sum(len(v) for v in aliases.values())
            print(f"[deps] Loaded {n_alias} alias roots from tsconfig", file=sys.stderr, flush=True)

        print(f"[deps] Parsing {N} files...", file=sys.stderr, flush=True)

        edges: list[tuple[int, int]] = []
        for i, fpath in enumerate(files):
            try:
                text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                for raw in _extract_raw(fpath, text):
                    resolved = _resolve(raw, fpath, file_set, aliases)
                    if resolved and resolved in local_file_idx and resolved != fpath:
                        edges.append((i, local_file_idx[resolved]))
            except Exception:
                pass

        adj = self._build_sparse(N, edges)
        vram_kb = round((adj._nnz() * 3 * 4) / 1024, 1)
        print(f"[deps] {N} files, {len(edges)} edges, ~{vram_kb} KB sparse VRAM", file=sys.stderr, flush=True)
        with self._lock:
            self._files = files
            self._file_idx = local_file_idx
            self._edges = edges
            self._adj = adj
            if not append:
                self.base_dir = directory
        return {"files": N, "edges": len(edges)}

    def update_file(self, fpath: str):
        """Re-parse one file's imports and rebuild the sparse matrix."""
        fpath = os.path.abspath(fpath)
        with self._lock:
            if self._adj is None or fpath not in self._file_idx:
                return
            i = self._file_idx[fpath]
            edges = [(s, t) for s, t in self._edges if s != i]
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
                    edges.append((i, local_file_idx[resolved]))
        except Exception:
            pass
        adj = self._build_sparse(n_files, edges)
        with self._lock:
            self._edges = edges
            self._adj = adj

    def impact(self, fpath: str, max_hops: int = 20) -> list[dict]:
        """GPU BFS over sparse adjacency: find all files that transitively import fpath."""
        fpath = os.path.abspath(fpath)
        with self._lock:
            if self._adj is None or fpath not in self._file_idx:
                return []
            target = self._file_idx[fpath]
            N = len(self._files)
            adj_snapshot = self._adj
            files_snapshot = list(self._files)

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
                results.append({"file": files_snapshot[idx], "hops": hop})
            frontier = (torch.sparse.mm(adj_snapshot, new.unsqueeze(1)).squeeze(1) > 0).float()
            frontier[target] = 0.0

        return results

    def direct_imports(self, fpath: str) -> list[str]:
        """Return files that fpath directly imports."""
        fpath = os.path.abspath(fpath)
        with self._lock:
            if self._adj is None or fpath not in self._file_idx:
                return []
            i = self._file_idx[fpath]
            indices = self._adj.indices()
            row_mask = indices[0] == i
            return [self._files[j] for j in indices[1][row_mask].cpu().tolist()]

    def stats(self) -> dict:
        with self._lock:
            N = len(self._files)
            edges = self._adj._nnz() if self._adj is not None else 0
            base_dir = self.base_dir
        return {"files": N, "edges": edges, "base_dir": base_dir}
