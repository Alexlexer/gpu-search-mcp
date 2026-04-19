import os
from pathlib import Path
from typing import Optional
import torch
import numpy as np


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _best_device()

INDEXED_EXTS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.c', '.cpp', '.h',
    '.hpp', '.java', '.cs', '.rb', '.php', '.swift', '.kt', '.json', '.yaml',
    '.yml', '.toml', '.md', '.txt', '.html', '.css', '.scss', '.sql', '.sh',
    '.bat', '.ps1', '.cfg', '.ini', '.xml', '.env'
}

SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build',
    '.next', '.nuxt', 'target', 'bin', 'obj', '.idea', '.vscode', '.mypy_cache'
}

# Null-byte separator between files — prevents cross-file false matches
_SEP = b'\x00'
_SEP_LEN = len(_SEP)


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
        self._file_starts: Optional[torch.Tensor] = None   # (N+1,) byte start of each file
        self._file_names: list[str] = []
        # Per-file newlines stored as one flat CPU array with an offset index
        self._nl_data: Optional[np.ndarray] = None         # concatenated newline positions (file-relative)
        self._nl_starts: Optional[np.ndarray] = None       # (N+1,) index into _nl_data per file
        self._vram_bytes = 0
        self.base_dir: Optional[str] = None

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
        corpus_t = torch.from_numpy(corpus_np).to(DEVICE)
        self._corpus_raw = corpus_t
        self._corpus_lower = _to_lower(corpus_t)
        self._file_starts = torch.tensor(file_starts_list, dtype=torch.long, device=DEVICE)
        self._nl_data = np.concatenate(nl_data_list).astype(np.int32) if nl_data_list else np.array([], np.int32)
        self._nl_starts = np.array(nl_starts_list, dtype=np.int64)
        self._vram_bytes = corpus_t.nbytes * 2 + self._file_starts.nbytes

    def index_directory(self, directory: str, max_file_mb: float = 5.0, append: bool = False) -> dict:
        directory = os.path.abspath(directory)
        max_bytes = int(max_file_mb * 1024 * 1024)

        existing: list[tuple[str, bytes]] = []
        if append and self._file_names:
            # Re-read existing files not from this directory (keep them)
            for name in self._file_names:
                if not name.startswith(directory):
                    try:
                        existing.append((name, open(name, 'rb').read()))
                    except Exception:
                        pass
        else:
            self.base_dir = directory

        new_files: list[tuple[str, bytes]] = []
        indexed = skipped = 0

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if Path(fname).suffix.lower() not in INDEXED_EXTS:
                    skipped += 1
                    continue
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                    if size == 0 or size > max_bytes:
                        skipped += 1
                        continue
                    raw = open(fpath, 'rb').read()
                    new_files.append((fpath, raw))
                    indexed += 1
                except Exception:
                    skipped += 1

        all_files = existing + new_files
        self._file_names = [f for f, _ in all_files]
        self._build_corpus(all_files)

        return {
            'indexed': indexed,
            'skipped': skipped,
            'vram_mb': round(self._vram_bytes / 1024 / 1024, 2),
        }

    def update_file(self, fpath: str):
        """Re-index a single file by rebuilding the corpus."""
        fpath = os.path.abspath(fpath)
        if self._corpus_raw is None:
            return
        if Path(fpath).suffix.lower() not in INDEXED_EXTS:
            return

        # Rebuild file list with this file updated/removed
        new_list: list[tuple[str, bytes]] = []
        for name in self._file_names:
            if name == fpath:
                continue  # will re-add below if exists
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

    def search(self, pattern: str, case_sensitive: bool = False,
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

        # Decode lines for matched files — read from disk (10-50 files, negligible latency)
        results = []

        for fi, local_hits in seen_files.items():
            fpath = self._file_names[fi]
            try:
                raw_file = np.frombuffer(open(fpath, 'rb').read(), dtype=np.uint8)
            except Exception:
                continue

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
        }
        if DEVICE.type == "cuda":
            result['vram_total_mb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
            result['vram_reserved_mb'] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2)
        return result
