"""Tests for semantic cache metadata validation and stale cache rejection."""
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

import gpu_semantic_index as gsi
from cache_manager import (
    SEMANTIC_CACHE_SCHEMA_VERSION,
    compute_source_fingerprint,
    upsert_cache_entry,
)
from gpu_index import SKIP_DIRS
from gpu_semantic_index import (
    CHUNK_LINES,
    MODEL_ID,
    OVERLAP_LINES,
    SemanticIndex,
    _SEMANTIC_EXTS,
    _cache_path,
)
from server_config import VERSION


def _write_cache(directory: str, meta_override: dict = None, embed_dim: int = 4):
    """Write a minimal valid cache for testing."""
    chunks = [{"file": os.path.join(directory, "f.py"), "start_line": 1, "end_line": 5, "text": "hello world"}]
    embeddings = np.random.randn(1, embed_dim).astype(np.float32)
    fp = gsi._dir_fingerprint(directory, 5.0)
    meta = {
        "model_id": MODEL_ID,
        "chunk_lines": CHUNK_LINES,
        "overlap_lines": OVERLAP_LINES,
        "directory": directory,
        "fingerprint": fp,
        "embed_dim": embed_dim,
        "created": "2026-01-01T00:00:00Z",
    }
    if meta_override:
        meta.update(meta_override)
    cache = _cache_path(directory)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        metadata_json=np.array(json.dumps(meta)),
        chunks_json=np.array(json.dumps(chunks)),
        embeddings=embeddings,
    )
    source_fingerprint = compute_source_fingerprint(
        directory,
        _SEMANTIC_EXTS,
        SKIP_DIRS,
        settings={
            "cache": "semantic",
            "model_id": MODEL_ID,
            "chunk_lines": CHUNK_LINES,
            "overlap_lines": OVERLAP_LINES,
        },
    )
    upsert_cache_entry(
        cache.parent,
        directory,
        VERSION,
        name="semantic",
        schema_version=SEMANTIC_CACHE_SCHEMA_VERSION,
        file_path=cache,
        source_fingerprint=source_fingerprint,
        status="rebuilt",
    )
    return cache


class TestCacheMetadataValidation:
    def test_valid_cache_loads(self, tmp_path):
        (tmp_path / "f.py").write_text("x = 1\n")
        _write_cache(str(tmp_path))
        idx = SemanticIndex()
        ok = idx._load_cache(str(tmp_path))
        assert ok, "Valid cache should load successfully"
        assert len(idx._chunks) == 1

    def test_stale_cache_rejected_on_model_id_change(self, tmp_path):
        (tmp_path / "f.py").write_text("x = 1\n")
        _write_cache(str(tmp_path), meta_override={"model_id": "old/model-v0"})
        idx = SemanticIndex()
        ok = idx._load_cache(str(tmp_path))
        assert not ok, "Cache with wrong model_id should be rejected"
        assert not _cache_path(str(tmp_path)).exists(), "Stale cache file should be deleted"

    def test_stale_cache_rejected_on_chunk_lines_change(self, tmp_path):
        (tmp_path / "f.py").write_text("x = 1\n")
        _write_cache(str(tmp_path), meta_override={"chunk_lines": CHUNK_LINES + 10})
        idx = SemanticIndex()
        ok = idx._load_cache(str(tmp_path))
        assert not ok, "Cache with wrong chunk_lines should be rejected"

    def test_stale_cache_rejected_on_fingerprint_change(self, tmp_path):
        (tmp_path / "f.py").write_text("x = 1\n")
        _write_cache(str(tmp_path), meta_override={"fingerprint": "deadbeefcafe"})
        idx = SemanticIndex()
        ok = idx._load_cache(str(tmp_path))
        assert not ok, "Cache with wrong fingerprint should be rejected"

    def test_old_format_cache_rejected(self, tmp_path):
        """Cache without metadata_json key (old format) should be rejected and deleted."""
        (tmp_path / "f.py").write_text("x = 1\n")
        cache = _cache_path(str(tmp_path))
        chunks = [{"file": str(tmp_path / "f.py"), "start_line": 1, "end_line": 5, "text": "hello"}]
        embeddings = np.random.randn(1, 4).astype(np.float32)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, chunks_json=np.array(json.dumps(chunks)), embeddings=embeddings)
        assert cache.exists()

        idx = SemanticIndex()
        ok = idx._load_cache(str(tmp_path))
        assert not ok, "Old-format cache (no metadata) should be rejected"
        assert not cache.exists(), "Old-format cache should be deleted"

    def test_corrupt_cache_rejected(self, tmp_path):
        (tmp_path / "f.py").write_text("x = 1\n")
        cache = _cache_path(str(tmp_path))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"not a numpy file")
        idx = SemanticIndex()
        ok = idx._load_cache(str(tmp_path))
        assert not ok

    def test_save_then_load_roundtrip(self, tmp_path):
        """_save_cache followed by _load_cache should succeed for the same directory."""
        (tmp_path / "f.py").write_text("x = 1\n")

        # Build a minimal in-memory state and save it
        import torch
        idx = SemanticIndex()
        idx._chunks = [{"file": str(tmp_path / "f.py"), "start_line": 1, "end_line": 5, "text": "x = 1"}]
        idx._embeddings = torch.zeros(1, 4)
        idx._vram_bytes = idx._embeddings.nbytes
        idx.base_dir = str(tmp_path)
        idx._save_cache(str(tmp_path))

        assert _cache_path(str(tmp_path)).exists()

        idx2 = SemanticIndex()
        ok = idx2._load_cache(str(tmp_path))
        assert ok
        assert len(idx2._chunks) == 1
