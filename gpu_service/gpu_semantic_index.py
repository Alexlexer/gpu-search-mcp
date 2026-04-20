import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional
import torch
import numpy as np

from gpu_index import INDEXED_EXTS, SKIP_DIRS, _best_device

DEVICE = _best_device()
MODEL_ID = "BAAI/bge-small-en-v1.5"
CHUNK_LINES = 40
OVERLAP_LINES = 8
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOC_PREFIX = ""

# Embed on CUDA when available — ~10x faster than CPU for large repos.
# MPS excluded: sentence-transformers MPS support is inconsistent.
_EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256 if _EMBED_DEVICE == "cuda" else 64

# Semantic search adds value for code, not for data/config files.
_SEMANTIC_EXTS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.c', '.cpp', '.h',
    '.hpp', '.java', '.cs', '.rb', '.php', '.swift', '.kt', '.sh', '.sql',
    '.md', '.txt',
}


def _chunk_file(fpath: str, text: str) -> list[dict]:
    lines = text.splitlines()
    chunks = []
    step = CHUNK_LINES - OVERLAP_LINES
    for start in range(0, max(1, len(lines)), step):
        end = min(start + CHUNK_LINES, len(lines))
        content = "\n".join(lines[start:end])
        if content.strip():
            chunks.append({
                "file": fpath,
                "start_line": start + 1,
                "end_line": end,
                "text": content,
            })
        if end >= len(lines):
            break
    return chunks


def _dir_fingerprint(directory: str, max_file_mb: float) -> str:
    """Hash of (filepath, mtime, size) for all indexed files — detects changes."""
    max_bytes = int(max_file_mb * 1024 * 1024)
    entries = []
    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fname in sorted(files):
            if Path(fname).suffix.lower() not in _SEMANTIC_EXTS:
                continue
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
                if st.st_size == 0 or st.st_size > max_bytes:
                    continue
                entries.append(f"{fpath}:{st.st_mtime}:{st.st_size}")
            except Exception:
                pass
    return hashlib.md5("\n".join(entries).encode()).hexdigest()


def _cache_path(directory: str) -> Path:
    safe = directory.replace(":", "").replace("\\", "_").replace("/", "_").strip("_")
    return Path(os.path.dirname(__file__)) / f".semantic_cache_{safe}.npz"


class SemanticIndex:
    def __init__(self):
        self._model = None
        self._embeddings: Optional[torch.Tensor] = None
        self._chunks: list[dict] = []
        self._vram_bytes = 0
        self.base_dir: Optional[str] = None
        self._embed_status: str = ""
        self._last_error: str = ""

    def _get_model(self):
        if self._model is None:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            from sentence_transformers import SentenceTransformer
            print(f"[semantic] Loading {MODEL_ID} on {_EMBED_DEVICE}...", file=sys.stderr, flush=True)
            self._model = SentenceTransformer(MODEL_ID, device=_EMBED_DEVICE)
            self._model.max_seq_length = 256  # code chunks are dense; truncating saves 2-4x tokenization time
            print(f"[semantic] Model ready", file=sys.stderr, flush=True)
        return self._model

    def _load_cache(self, directory: str) -> bool:
        cache = _cache_path(directory)
        if not cache.exists():
            return False
        try:
            data = np.load(cache, allow_pickle=True)
            chunks = json.loads(str(data["chunks_json"]))
            embeddings = torch.from_numpy(data["embeddings"]).to(DEVICE)
            if self._embeddings is not None:
                self._vram_bytes -= self._embeddings.nbytes
            self._chunks = chunks
            self._embeddings = embeddings
            self._vram_bytes = embeddings.nbytes
            self.base_dir = directory
            print(f"[semantic] Loaded {len(chunks)} chunks from cache", file=sys.stderr, flush=True)
            return True
        except Exception as e:
            print(f"[semantic] Cache load failed: {e}", file=sys.stderr, flush=True)
            return False

    def _save_cache(self, directory: str):
        cache = _cache_path(directory)
        try:
            np.savez(
                cache,
                chunks_json=np.array(json.dumps(self._chunks)),
                embeddings=self._embeddings.cpu().numpy(),
            )
            print(f"[semantic] Cache saved to {cache.name}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[semantic] Cache save failed: {e}", file=sys.stderr, flush=True)

    def try_load_cache(self, directory: str, max_file_mb: float = 5.0) -> Optional[dict]:
        """Load from cache if it exists — no model needed. Returns stats or None."""
        directory = os.path.abspath(directory)
        if self._load_cache(directory):
            return {"chunks": len(self._chunks), "vram_mb": round(self._vram_bytes / 1024 / 1024, 2)}
        return None

    def merge_cache(self, directory: str) -> Optional[dict]:
        """Append another directory's cache into the existing index without replacing it."""
        directory = os.path.abspath(directory)
        cache = _cache_path(directory)
        if not cache.exists():
            return None
        try:
            data = np.load(cache, allow_pickle=True)
            new_chunks = json.loads(str(data["chunks_json"]))
            new_embs = torch.from_numpy(data["embeddings"]).to(DEVICE)
            if self._embeddings is None:
                self._chunks = new_chunks
                self._embeddings = new_embs
                self.base_dir = directory
            else:
                self._chunks = self._chunks + new_chunks
                self._embeddings = torch.cat([self._embeddings, new_embs], dim=0)
            self._vram_bytes = self._embeddings.nbytes
            print(f"[semantic] Merged {len(new_chunks)} chunks from {os.path.basename(directory)}", file=sys.stderr, flush=True)
            return {"chunks": len(self._chunks), "vram_mb": round(self._vram_bytes / 1024 / 1024, 2)}
        except Exception as e:
            print(f"[semantic] merge_cache failed for {directory}: {e}", file=sys.stderr, flush=True)
            return None

    def _embed(self, texts: list[str]) -> torch.Tensor:
        import time
        model = self._get_model()
        all_embs = []
        t0 = time.time()
        n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        for bi, i in enumerate(range(0, len(texts), BATCH_SIZE)):
            batch = texts[i: i + BATCH_SIZE]
            with torch.no_grad():
                emb = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(emb.to(DEVICE))
            if bi == 0 or (bi + 1) % 10 == 0 or bi + 1 == n_batches:
                elapsed = time.time() - t0
                rate = (bi + 1) * BATCH_SIZE / elapsed
                eta = (n_batches - bi - 1) * BATCH_SIZE / rate if rate > 0 else 0
                self._embed_status = f"batch {bi+1}/{n_batches}, {rate:.0f} chunks/s, ETA {eta:.0f}s"
                print(f"[semantic] {self._embed_status}", file=sys.stderr, flush=True)
        return torch.cat(all_embs, dim=0)

    def index_directory(self, directory: str, max_file_mb: float = 5.0, append: bool = False, force: bool = False) -> dict:
        directory = os.path.abspath(directory)
        max_bytes = int(max_file_mb * 1024 * 1024)

        if not append and not force and self._load_cache(directory):
            self.base_dir = directory
            return {"chunks": len(self._chunks), "skipped": 0, "vram_mb": round(self._vram_bytes / 1024 / 1024, 2), "from_cache": True}

        # Cache miss — walk, chunk, embed
        chunks: list[dict] = []
        skipped = 0
        print(f"[semantic] Cache miss — walking {directory}", file=sys.stderr, flush=True)
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fname in files:
                if Path(fname).suffix.lower() not in _SEMANTIC_EXTS:
                    skipped += 1
                    continue
                fpath = os.path.join(root, fname)
                try:
                    size = os.path.getsize(fpath)
                    if size == 0 or size > max_bytes:
                        skipped += 1
                        continue
                    text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                    chunks.extend(_chunk_file(fpath, text))
                except Exception as e:
                    print(f"[semantic] Skipped {fname}: {e}", file=sys.stderr, flush=True)
                    skipped += 1

        print(f"[semantic] {len(chunks)} chunks, {skipped} skipped. Embedding...", file=sys.stderr, flush=True)

        if not chunks:
            if not append:
                self._chunks = []
                self._embeddings = None
                self.base_dir = directory
            return {"chunks": 0, "skipped": skipped, "vram_mb": 0.0}

        try:
            embeddings = self._embed([DOC_PREFIX + c["text"] for c in chunks])
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self._last_error = f"embed failed: {e}\n{tb}"
            print(f"[semantic] EMBED FAILED: {e}\n{tb}", file=sys.stderr, flush=True)
            raise
        print(f"[semantic] {len(chunks)} chunks embedded", file=sys.stderr, flush=True)

        if append and self._embeddings is not None:
            # Remove old chunks from this directory, then append new ones
            keep = [i for i, c in enumerate(self._chunks) if not c["file"].startswith(directory)]
            kept_chunks = [self._chunks[i] for i in keep]
            kept_embs = self._embeddings[torch.tensor(keep, device=DEVICE)] if keep else torch.zeros((0, embeddings.shape[1]), device=DEVICE)
            self._chunks = kept_chunks + chunks
            self._embeddings = torch.cat([kept_embs, embeddings], dim=0)
        else:
            self._chunks = chunks
            self._embeddings = embeddings
            self.base_dir = directory

        self._vram_bytes = self._embeddings.nbytes
        self._save_cache(self.base_dir or directory)
        return {"chunks": len(chunks), "skipped": skipped, "vram_mb": round(self._vram_bytes / 1024 / 1024, 2)}

    def update_file(self, fpath: str):
        """Incrementally re-embed a single changed file. Called by watchdog."""
        fpath = os.path.abspath(fpath)
        if self._embeddings is None or Path(fpath).suffix.lower() not in _SEMANTIC_EXTS:
            return
        try:
            # Drop old chunks for this file
            keep_idx = [i for i, c in enumerate(self._chunks) if c["file"] != fpath]
            kept_chunks = [self._chunks[i] for i in keep_idx]
            kept_embs = self._embeddings[torch.tensor(keep_idx, device=DEVICE)] if keep_idx else torch.zeros((0, self._embeddings.shape[1]), device=DEVICE)

            # Re-chunk and re-embed
            new_chunks: list[dict] = []
            if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                new_chunks = _chunk_file(fpath, text)

            if new_chunks:
                new_embs = self._embed([DOC_PREFIX + c["text"] for c in new_chunks])
                self._chunks = kept_chunks + new_chunks
                self._embeddings = torch.cat([kept_embs, new_embs], dim=0)
            else:
                self._chunks = kept_chunks
                self._embeddings = kept_embs if kept_embs.shape[0] > 0 else None

            if self._embeddings is not None:
                self._vram_bytes = self._embeddings.nbytes
                if self.base_dir:
                    self._save_cache(self.base_dir)
            print(f"[semantic] Updated {os.path.basename(fpath)}: {len(new_chunks)} chunks", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[semantic] update_file failed for {fpath}: {e}", file=sys.stderr, flush=True)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._embeddings is None or not self._chunks:
            return []
        model = self._get_model()
        with torch.no_grad():
            q_emb = model.encode(
                [QUERY_PREFIX + query],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).to(DEVICE)
        scores = (q_emb @ self._embeddings.T).squeeze(0)
        k = min(top_k, len(self._chunks))
        top_scores, top_idx = torch.topk(scores, k)
        results = []
        for score, idx in zip(top_scores.cpu().tolist(), top_idx.cpu().tolist()):
            chunk = self._chunks[idx]
            results.append({
                "file": chunk["file"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "score": round(score, 4),
                "snippet": chunk["text"][:400],
            })
        return results

    def stats(self) -> dict:
        d = {
            "chunks": len(self._chunks),
            "vram_mb": round(self._vram_bytes / 1024 / 1024, 2),
            "base_dir": self.base_dir,
        }
        if self._embed_status:
            d["embed_progress"] = self._embed_status
        if self._last_error:
            d["last_error"] = self._last_error
        return d
