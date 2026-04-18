import os
import sys
from pathlib import Path
from typing import Optional
import torch

from gpu_index import INDEXED_EXTS, SKIP_DIRS

DEVICE = torch.device("cuda")
MODEL_ID = "BAAI/bge-small-en-v1.5"
CHUNK_LINES = 40
OVERLAP_LINES = 8
BATCH_SIZE = 64
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DOC_PREFIX = ""


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


class SemanticIndex:
    def __init__(self):
        self._model = None
        self._embeddings: Optional[torch.Tensor] = None  # (N, D) float32 on CUDA, normalized
        self._chunks: list[dict] = []
        self._vram_bytes = 0
        self.base_dir: Optional[str] = None

    def _get_model(self):
        if self._model is None:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

            from sentence_transformers import SentenceTransformer
            print(f"[semantic] Downloading/loading {MODEL_ID} (first run: ~500MB)...", file=sys.stderr, flush=True)
            self._model = SentenceTransformer(
                MODEL_ID,
                device="cuda",
            )
            print(f"[semantic] Model loaded onto GPU", file=sys.stderr, flush=True)
        return self._model

    def index_directory(self, directory: str, max_file_mb: float = 5.0) -> dict:
        directory = os.path.abspath(directory)
        self.base_dir = directory
        max_bytes = int(max_file_mb * 1024 * 1024)
        chunks: list[dict] = []
        skipped = 0

        print(f"[semantic] Walking {directory}", file=sys.stderr, flush=True)
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
                    text = Path(fpath).read_text(encoding="utf-8", errors="replace")
                    file_chunks = _chunk_file(fpath, text)
                    chunks.extend(file_chunks)
                    print(f"[semantic] Chunked {fname} → {len(file_chunks)} chunks", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"[semantic] Skipped {fname}: {e}", file=sys.stderr, flush=True)
                    skipped += 1

        print(f"[semantic] Total: {len(chunks)} chunks, {skipped} skipped", file=sys.stderr, flush=True)

        if not chunks:
            self._chunks = []
            self._embeddings = None
            return {"chunks": 0, "skipped": skipped, "vram_mb": 0.0}

        print("[semantic] Loading model...", file=sys.stderr, flush=True)
        model = self._get_model()
        print("[semantic] Model ready. Embedding chunks...", file=sys.stderr, flush=True)
        texts = [DOC_PREFIX + c["text"] for c in chunks]

        all_embs: list[torch.Tensor] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            print(f"[semantic] Batch {i//BATCH_SIZE + 1}/{-(-len(texts)//BATCH_SIZE)} ({len(batch)} chunks)...", file=sys.stderr, flush=True)
            with torch.no_grad():
                emb = model.encode(
                    batch,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            all_embs.append(emb.to(DEVICE))

        embeddings = torch.cat(all_embs, dim=0)  # (N, D) on CUDA

        if self._embeddings is not None:
            self._vram_bytes -= self._embeddings.nbytes

        self._chunks = chunks
        self._embeddings = embeddings
        self._vram_bytes = embeddings.nbytes

        return {
            "chunks": len(chunks),
            "skipped": skipped,
            "vram_mb": round(self._vram_bytes / 1024 / 1024, 2),
        }

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
            ).to(DEVICE)  # (1, D)

        scores = (q_emb @ self._embeddings.T).squeeze(0)  # (N,)
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
        return {
            "chunks": len(self._chunks),
            "vram_mb": round(self._vram_bytes / 1024 / 1024, 2),
            "base_dir": self.base_dir,
        }
