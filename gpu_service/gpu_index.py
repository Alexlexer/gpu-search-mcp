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


def _to_lower(t: torch.Tensor) -> torch.Tensor:
    """ASCII lowercase using vectorized GPU ops."""
    lower = t.clone()
    mask = (t >= ord('A')) & (t <= ord('Z'))
    lower[mask] += 32
    return lower


def _search(data: torch.Tensor, pattern_bytes: bytes) -> torch.Tensor:
    """
    GPU pattern search: first-char filter then vectorized window verification.
    No custom kernels — pure PyTorch ops on RTX VRAM.
    """
    n, m = len(data), len(pattern_bytes)
    if n == 0 or m == 0 or n < m:
        return torch.tensor([], dtype=torch.long, device=DEVICE)

    pat = torch.tensor(list(pattern_bytes), dtype=torch.uint8, device=DEVICE)

    # Only consider positions where first char matches
    candidates = (data[:n - m + 1] == pat[0]).nonzero(as_tuple=True)[0]
    if len(candidates) == 0:
        return torch.tensor([], dtype=torch.long, device=DEVICE)
    if m == 1:
        return candidates

    # Build (C, m) index matrix and compare all chars at once
    offsets = torch.arange(m, device=DEVICE)
    indices = candidates.unsqueeze(1) + offsets.unsqueeze(0)  # (C, m)
    chars = data[indices]                                       # (C, m)
    match_mask = (chars == pat).all(dim=1)
    return candidates[match_mask]


class GpuFileIndex:
    def __init__(self):
        # fpath -> (raw_gpu, lower_gpu, newlines_gpu)
        self._files: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._vram_bytes = 0
        self.base_dir: Optional[str] = None

    def index_directory(self, directory: str, max_file_mb: float = 5.0, append: bool = False) -> dict:
        directory = os.path.abspath(directory)
        if not append:
            self._files = {}
            self._vram_bytes = 0
        self.base_dir = directory if not append else (self.base_dir or directory)
        max_bytes = int(max_file_mb * 1024 * 1024)
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
                    self._load_file(fpath)
                    indexed += 1
                except Exception:
                    skipped += 1

        return {
            'indexed': indexed,
            'skipped': skipped,
            'vram_mb': round(self._vram_bytes / 1024 / 1024, 2),
        }

    def _load_file(self, fpath: str):
        with open(fpath, 'rb') as f:
            raw_bytes = f.read()
        if not raw_bytes:
            return

        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        raw = torch.from_numpy(arr.copy()).to(DEVICE)
        lower = _to_lower(raw)
        newlines = (raw == ord('\n')).nonzero(as_tuple=True)[0]

        if fpath in self._files:
            old = self._files[fpath]
            self._vram_bytes -= sum(t.nbytes for t in old)

        self._files[fpath] = (raw, lower, newlines)
        self._vram_bytes += raw.nbytes + lower.nbytes + newlines.nbytes

    def update_file(self, fpath: str):
        fpath = os.path.abspath(fpath)
        if not os.path.exists(fpath):
            if fpath in self._files:
                old = self._files.pop(fpath)
                self._vram_bytes -= sum(t.nbytes for t in old)
            return
        if Path(fpath).suffix.lower() not in INDEXED_EXTS:
            return
        try:
            self._load_file(fpath)
        except Exception:
            pass

    def search(self, pattern: str, case_sensitive: bool = False,
               max_files: int = 50) -> list[dict]:
        if not pattern or not self._files:
            return []

        pat_bytes = pattern.encode('utf-8', errors='replace')
        if not case_sensitive:
            pat_bytes = pat_bytes.lower()

        # Phase 1: GPU-only pass — find all matching files without touching CPU
        matching: list[tuple[str, torch.Tensor]] = []
        total_files_scanned = 0
        for fpath, (raw, lower, newlines) in self._files.items():
            total_files_scanned += 1
            search_data = lower if not case_sensitive else raw
            positions = _search(search_data, pat_bytes)
            if len(positions) > 0:
                matching.append((fpath, positions))

        # Phase 2: CPU decode — only for matched files, capped at max_files
        results = []
        total_match_files = len(matching)
        for fpath, positions in matching[:max_files]:
            _, lower, newlines = self._files[fpath]
            raw = self._files[fpath][0]
            line_nos = torch.searchsorted(newlines.cpu(), positions.cpu()).numpy()
            pos_cpu = positions.cpu().numpy()
            nl_cpu = newlines.cpu().numpy()
            raw_cpu = raw.cpu().numpy()

            seen_lines: set[int] = set()
            matches = []
            for pos, line_no in zip(pos_cpu, line_nos):
                ln = int(line_no)
                if ln in seen_lines:
                    continue
                seen_lines.add(ln)
                start = int(nl_cpu[ln - 1]) + 1 if ln > 0 else 0
                end = int(nl_cpu[ln]) if ln < len(nl_cpu) else len(raw_cpu)
                content = raw_cpu[start:end].tobytes().decode('utf-8', errors='replace').rstrip()
                matches.append({'line': ln + 1, 'content': content})
                if len(matches) >= 10:
                    break

            results.append({'file': fpath, 'matches': matches})

        # Attach total count for the summary line
        for r in results:
            r['_total_files'] = total_match_files
        return results

    def stats(self) -> dict:
        result = {
            'files': len(self._files),
            'vram_mb': round(self._vram_bytes / 1024 / 1024, 2),
            'base_dir': self.base_dir,
        }
        if DEVICE.type == "cuda":
            result['vram_total_mb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
            result['vram_reserved_mb'] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 2)
        return result
