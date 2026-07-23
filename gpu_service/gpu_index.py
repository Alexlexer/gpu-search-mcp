import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional
import torch
import numpy as np

from cache_manager import (
    PATTERN_CACHE_SCHEMA_VERSION,
    cache_transaction,
    compute_source_fingerprint,
    invalidate_cache_entry,
    is_cache_entry_valid,
    load_cache_metadata,
    upsert_cache_entry,
)
from server_config import VERSION

from device import DeviceInfo, resolve_torch_device  # noqa: E402 (after torch)

DEVICE_INFO: DeviceInfo = resolve_torch_device(os.environ.get("GPU_SEARCH_DEVICE"))
DEVICE = torch.device(DEVICE_INFO.torch_device)


def _best_device() -> torch.device:
    """Backward-compat shim — returns the already-resolved DEVICE."""
    return DEVICE

def _file_ext(fname: str) -> str:
    """Return the indexable extension for a filename.

    Handles dotfiles like .env whose pathlib suffix is '' by treating
    the whole name (with leading dot) as the extension.
    """
    p = Path(fname)
    ext = p.suffix.lower()
    if not ext and p.name.startswith('.') and p.name.count('.') == 1:
        # e.g. '.env', '.gitignore' — name IS the extension
        ext = p.name.lower()
    return ext


INDEXED_EXTS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.c', '.cpp', '.h',
    '.hpp', '.java', '.cs', '.rb', '.php', '.swift', '.kt', '.json', '.yaml',
    '.yml', '.toml', '.md', '.txt', '.html', '.css', '.scss', '.sql', '.sh',
    '.bat', '.ps1', '.cfg', '.ini', '.xml',
    # .env is excluded by default — use allow_env_files=True in index_directory to opt in
}

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build',
    '.next', '.nuxt', 'target', 'bin', 'obj', '.idea', '.vscode', '.mypy_cache',
    '.gpu-search-cache',
}

# Null-byte separator between files — prevents cross-file false matches
_SEP = b'\x00'
_SEP_LEN = len(_SEP)


def _pattern_cache_components(allow_env_files: bool) -> dict:
    return {
        "parser": "byte-pattern-v1",
        "lineOffsets": "int32-newline-v1",
        "allowEnvFiles": allow_env_files,
    }


def _to_lower(t: torch.Tensor) -> torch.Tensor:
    lower = t.clone()
    mask = (t >= ord('A')) & (t <= ord('Z'))
    lower[mask] += 32
    return lower


class GpuFileIndex:
    """
    Single-corpus architecture: all files concatenated into one VRAM tensor.
    Search is a single GPU kernel pass with one sync point — no per-file loop.
    """

    def __init__(self):
        self._corpus_raw: Optional[torch.Tensor] = None    # (total_bytes,) uint8
        self._corpus_lower: Optional[torch.Tensor] = None  # lowercase corpus
        self._corpus_raw_cpu: Optional[np.ndarray] = None  # CPU mirror — avoids re-download per search
        self._file_starts: Optional[torch.Tensor] = None   # (N+1,) byte start of each file
        self._file_names: list[str] = []
        # Per-file newlines stored as one flat CPU array with an offset index
        self._nl_data: Optional[np.ndarray] = None         # concatenated newline positions (file-relative)
        self._nl_starts: Optional[np.ndarray] = None       # (N+1,) index into _nl_data per file
        self._vram_bytes = 0
        self.base_dir: Optional[str] = None
        self._file_meta: dict[str, dict] = {}
        self._cache_status = "cold"
        self._lock = threading.Lock()

    def _cache_dir(self, directory: str) -> Path:
        return Path(directory) / ".gpu-search-cache"

    def _signature(self, fpath: str) -> Optional[dict]:
        try:
            st = os.stat(fpath)
            return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except OSError:
            return None

    def _hash_bytes(self, raw: bytes) -> str:
        return hashlib.blake2b(raw, digest_size=16).hexdigest()

    def _discover_files(self, directory: str, max_bytes: int, effective_exts: set[str]) -> tuple[list[str], int]:
        files: list[str] = []
        skipped = 0
        for root, dirs, fnames in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in fnames:
                ext = _file_ext(fname)
                if ext not in effective_exts:
                    skipped += 1
                    continue
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                    if size == 0 or size > max_bytes:
                        skipped += 1
                        continue
                    files.append(fpath)
                except Exception:
                    skipped += 1
        files.sort()
        return files, skipped

    def _load_pattern_cache(self, directory: str, discovered: list[str],
                            allow_env_files: bool) -> Optional[list[tuple[str, bytes]]]:
        cache_dir = self._cache_dir(directory)
        manifest_path = cache_dir / "cache-manifest.json"
        files_path = cache_dir / "files-v1.json"
        blob_path = cache_dir / "pattern-index-v1.bin"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("pattern_version") != 1 or manifest.get("allow_env_files") != allow_env_files:
                return None
            # Reject cache written for a different directory (cross-repo leakage guard).
            cached_dir = manifest.get("directory")
            if cached_dir and Path(cached_dir).resolve() != Path(directory).resolve():
                return None
            entries = json.loads(files_path.read_text(encoding="utf-8"))["files"]
            # Reject cache containing paths under .gpu-search-cache (stale/corrupted entry).
            if any(".gpu-search-cache" in Path(e["path"]).parts for e in entries):
                return None
            if [e["path"] for e in entries] != discovered:
                return None
            discovered_set = set(discovered)
            for e in entries:
                sig = self._signature(e["path"])
                if e["path"] not in discovered_set or sig is None:
                    return None
                if sig["size"] != e["size"] or sig["mtime_ns"] != e["mtime_ns"]:
                    return None
            blob = blob_path.read_bytes()
            file_list = []
            for e in entries:
                start = int(e["offset"])
                end = start + int(e["size"])
                raw = blob[start:end]
                if self._hash_bytes(raw) != e.get("hash"):
                    return None
                file_list.append((e["path"], raw))
            self._file_meta = {e["path"]: e for e in entries}
            self._cache_status = "loaded"
            return file_list
        except Exception:
            return None

    def _write_pattern_cache(self, directory: str, file_list: list[tuple[str, bytes]],
                             allow_env_files: bool, source_fingerprint: dict | None = None,
                             status: str = "rebuilt"):
        cache_dir = self._cache_dir(directory)
        try:
            entries = []
            offset = 0
            blobs = []
            line_offsets = []
            line_offset_start = 0
            for fpath, raw in file_list:
                sig = self._signature(fpath)
                if sig is None:
                    continue
                nls = np.where(np.frombuffer(raw, dtype=np.uint8) == ord('\n'))[0].astype(np.int32)
                entries.append({
                    "path": fpath,
                    "size": sig["size"],
                    "mtime_ns": sig["mtime_ns"],
                    "hash": self._hash_bytes(raw),
                    "offset": offset,
                    "line_offset_start": line_offset_start,
                    "line_count": int(len(nls)),
                })
                blobs.append(raw)
                line_offsets.append(nls.tobytes())
                offset += len(raw)
                line_offset_start += len(nls)
            fingerprint = source_fingerprint or compute_source_fingerprint(
                directory,
                INDEXED_EXTS | ({'.env'} if allow_env_files else set()),
                SKIP_DIRS,
                settings={"allow_env_files": allow_env_files, "cache": "pattern"},
            )
            manifest = {
                "pattern_version": 1,
                "directory": directory,
                "allow_env_files": allow_env_files,
                "file_count": len(entries),
                "updated_at": time.time(),
            }
            with cache_transaction(cache_dir, "pattern") as transaction:
                transaction.stage_bytes(
                    cache_dir / "pattern-index-v1.bin", b"".join(blobs)
                )
                transaction.stage_bytes(
                    cache_dir / "line-offsets-v1.bin", b"".join(line_offsets)
                )
                transaction.stage_json(
                    cache_dir / "files-v1.json", {"files": entries}
                )
                transaction.stage_json(cache_dir / "cache-manifest.json", manifest)
                upsert_cache_entry(
                    cache_dir,
                    directory,
                    VERSION,
                    name="pattern",
                    schema_version=PATTERN_CACHE_SCHEMA_VERSION,
                    file_path=cache_dir / "pattern-index-v1.bin",
                    source_fingerprint=fingerprint,
                    status=status,
                    components=_pattern_cache_components(allow_env_files),
                    transaction=transaction,
                )
            self._file_meta = {entry["path"]: entry for entry in entries}
        except Exception:
            pass
    def _build_corpus(self, file_list: list[tuple[str, bytes]]):
        """Concatenate file bytes into single GPU tensors."""
        chunks_raw = []
        file_starts_list = [0]
        nl_data_list = []
        nl_starts_list = [0]

        sep = np.frombuffer(_SEP, dtype=np.uint8)

        for _, raw_bytes in file_list:
            arr = np.frombuffer(raw_bytes, dtype=np.uint8)
            chunks_raw.append(arr)
            chunks_raw.append(sep)
            file_starts_list.append(file_starts_list[-1] + len(arr) + _SEP_LEN)
            # Newlines relative to start of this file
            nls = np.where(arr == ord('\n'))[0].astype(np.int32)
            nl_data_list.append(nls)
            nl_starts_list.append(nl_starts_list[-1] + len(nls))

        if not chunks_raw:
            return

        corpus_np = np.concatenate(chunks_raw)
        corpus_t = torch.from_numpy(corpus_np.copy()).to(DEVICE)
        self._corpus_raw = corpus_t
        self._corpus_lower = _to_lower(corpus_t)
        self._corpus_raw_cpu: Optional[np.ndarray] = corpus_np  # avoid re-downloading per search
        self._file_starts = torch.tensor(file_starts_list, dtype=torch.long, device=DEVICE)
        self._nl_data = np.concatenate(nl_data_list).astype(np.int32) if nl_data_list else np.array([], np.int32)
        self._nl_starts = np.array(nl_starts_list, dtype=np.int64)
        self._vram_bytes = corpus_t.nbytes * 2 + self._file_starts.nbytes

    def index_directory(
        self, directory: str, max_file_mb: float = 5.0,
        append: bool = False, allow_env_files: bool = False,
        force_rebuild: bool = False,
    ) -> dict:
        directory = os.path.abspath(directory)
        max_bytes = int(max_file_mb * 1024 * 1024)
        effective_exts = INDEXED_EXTS | ({'.env'} if allow_env_files else set())

        discovered, skipped = self._discover_files(directory, max_bytes, effective_exts)
        source_fingerprint = compute_source_fingerprint(
            directory,
            effective_exts,
            SKIP_DIRS,
            max_file_mb=max_file_mb,
            settings={"allow_env_files": allow_env_files, "cache": "pattern"},
        )
        metadata = load_cache_metadata(self._cache_dir(directory))
        entry_valid = is_cache_entry_valid(
            metadata,
            "pattern",
            PATTERN_CACHE_SCHEMA_VERSION,
            source_fingerprint,
            VERSION,
            _pattern_cache_components(allow_env_files),
        )
        if force_rebuild:
            invalidate_cache_entry(self._cache_dir(directory), "pattern", "rebuild_requested")
        elif metadata is not None and not entry_valid:
            invalidate_cache_entry(self._cache_dir(directory), "pattern", "stale")
        cached_files = None
        if not append and not force_rebuild and entry_valid:
            cached_files = self._load_pattern_cache(directory, discovered, allow_env_files)
        if cached_files is not None:
            new_files = cached_files
            indexed = len(new_files)
            cache_status = "loaded"
        else:
            new_files = []
            indexed = 0
            old_meta = dict(self._file_meta)
            cache_blob: dict[str, bytes] = {}
            if not append and old_meta:
                # Keep unchanged files from the previous in-memory corpus; changed files are read from disk.
                for name in self._file_names:
                    meta = old_meta.get(name)
                    sig = self._signature(name)
                    if meta and sig and sig["size"] == meta.get("size") and sig["mtime_ns"] == meta.get("mtime_ns"):
                        try:
                            cache_blob[name] = open(name, "rb").read()
                        except Exception:
                            pass
            for fpath in discovered:
                try:
                    raw = cache_blob.get(fpath)
                    if raw is None:
                        raw = open(fpath, 'rb').read()
                    new_files.append((fpath, raw))
                    indexed += 1
                except Exception:
                    skipped += 1
            cache_status = "rebuilt"

        with self._lock:
            existing: list[tuple[str, bytes]] = []
            if append and self._file_names:
                for name in self._file_names:
                    if not name.startswith(directory):
                        try:
                            existing.append((name, open(name, 'rb').read()))
                        except Exception:
                            pass
            else:
                self.base_dir = directory

            all_files = existing + new_files
            self._file_names = [f for f, _ in all_files]
            self._build_corpus(all_files)
            self._cache_status = cache_status

        if not append:
            self._write_pattern_cache(
                directory, new_files, allow_env_files, source_fingerprint, cache_status
            )

        return {
            'indexed': indexed,
            'skipped': skipped,
            'vram_mb': round(self._vram_bytes / 1024 / 1024, 2),
            'cache': self._cache_status,
        }

    def update_file(self, fpath: str, allow_env_files: bool = False):
        """Re-index a single file by rebuilding the corpus."""
        fpath = os.path.abspath(fpath)
        effective_exts = INDEXED_EXTS | ({'.env'} if allow_env_files else set())
        with self._lock:
            if self._corpus_raw is None:
                return
            if _file_ext(Path(fpath).name) not in effective_exts:
                return

            new_list: list[tuple[str, bytes]] = []
            for name in self._file_names:
                if name == fpath:
                    continue
                try:
                    new_list.append((name, open(name, 'rb').read()))
                except Exception:
                    pass

            if os.path.exists(fpath):
                try:
                    new_list.append((fpath, open(fpath, 'rb').read()))
                except Exception:
                    pass

            self._file_names = [f for f, _ in new_list]
            self._build_corpus(new_list)
            if self.base_dir:
                fingerprint = compute_source_fingerprint(
                    self.base_dir,
                    effective_exts,
                    SKIP_DIRS,
                    settings={"allow_env_files": allow_env_files, "cache": "pattern"},
                )
                self._write_pattern_cache(
                    self.base_dir, new_list, allow_env_files, fingerprint, "updated"
                )

    def search(self, pattern: str, case_sensitive: bool = False,
               max_files: int = 50) -> list[dict]:
        with self._lock:
            return self._search_locked(pattern, case_sensitive, max_files)

    def _search_locked(self, pattern: str, case_sensitive: bool = False,
                       max_files: int = 50) -> list[dict]:
        if not pattern or self._corpus_raw is None:
            return []

        pat_bytes = pattern.encode('utf-8', errors='replace')
        if not case_sensitive:
            pat_bytes = pat_bytes.lower()
        m = len(pat_bytes)

        corpus = self._corpus_lower if not case_sensitive else self._corpus_raw
        N_corpus = len(corpus)
        if N_corpus < m:
            return []

        # Single GPU pass across entire corpus
        pat_t = torch.tensor(list(pat_bytes), dtype=torch.uint8, device=DEVICE)
        offsets_t = torch.arange(m, device=DEVICE)

        candidates = (corpus[:N_corpus - m + 1] == pat_t[0]).nonzero(as_tuple=True)[0]
        if len(candidates) == 0:
            return []
        if m > 1:
            # Two-char pre-filter: cuts ~95% of false candidates before expensive matrix check
            c2_mask = corpus[candidates + 1] == pat_t[1]
            candidates = candidates[c2_mask]
            if len(candidates) == 0:
                return []
        if m > 2:
            idx_mat = candidates.unsqueeze(1) + offsets_t.unsqueeze(0)
            match_mask = (corpus[idx_mat] == pat_t).all(dim=1)
            candidates = candidates[match_mask]
        if len(candidates) == 0:
            return []

        # Map corpus positions → file indices (one searchsorted call)
        file_starts_cpu = self._file_starts.cpu()
        pos_cpu = candidates.cpu()
        file_indices = torch.searchsorted(file_starts_cpu, pos_cpu, right=True) - 1
        file_indices = file_indices.clamp(0, len(self._file_names) - 1)

        # Suppress matches that land in separator bytes
        match_local = pos_cpu - file_starts_cpu[file_indices]
        file_lens = file_starts_cpu[file_indices + 1] - file_starts_cpu[file_indices] - _SEP_LEN
        valid = match_local < file_lens
        pos_cpu = pos_cpu[valid]
        file_indices = file_indices[valid]
        match_local = match_local[valid]

        if len(pos_cpu) == 0:
            return []

        # Group by file, preserving order of first occurrence
        seen_files: dict[int, list[tuple[int, int]]] = {}
        for fi, local_p in zip(file_indices.tolist(), match_local.tolist()):
            fi = int(fi)
            if fi not in seen_files:
                if len(seen_files) >= max_files:
                    continue
                seen_files[fi] = []
            seen_files[fi].append((int(local_p),))

        total_match_files = len(
            set(file_indices.tolist())
        )

        # Decode lines for matched files (CPU only, small number of files)
        results = []
        raw_corpus_cpu = self._corpus_raw_cpu

        for fi, local_hits in seen_files.items():
            fpath = self._file_names[fi]
            f_start = int(file_starts_cpu[fi].item())
            f_end = int(file_starts_cpu[fi + 1].item()) - _SEP_LEN
            raw_file = raw_corpus_cpu[f_start:f_end]

            nl_s = int(self._nl_starts[fi])
            nl_e = int(self._nl_starts[fi + 1])
            nls = self._nl_data[nl_s:nl_e]  # newlines relative to file start

            seen_lines: set[int] = set()
            matches = []
            for (local_p,) in local_hits:
                ln = int(np.searchsorted(nls, local_p))
                if ln in seen_lines:
                    continue
                seen_lines.add(ln)
                line_start = int(nls[ln - 1]) + 1 if ln > 0 else 0
                line_end = int(nls[ln]) if ln < len(nls) else len(raw_file)
                content = raw_file[line_start:line_end].tobytes().decode('utf-8', errors='replace').rstrip()
                matches.append({'line': ln + 1, 'content': content})
                if len(matches) >= 10:
                    break

            results.append({'file': fpath, 'matches': matches, '_total_files': total_match_files})

        return results

    def stats(self) -> dict:
        result = {
            'files': len(self._file_names),
            'vram_mb': round(self._vram_bytes / 1024 / 1024, 2),
            'base_dir': self.base_dir,
            'cache': self._cache_status,
        }
        if DEVICE.type == "cuda":
            result['vram_total_mb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
            result['vram_reserved_mb'] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2)
        elif DEVICE.type == "mps":
            try:
                result['vram_allocated_mb'] = round(
                    torch.mps.current_allocated_memory() / 1024 / 1024, 2
                )
            except Exception:
                pass
        return result
