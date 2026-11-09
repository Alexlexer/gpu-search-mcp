"""Opt-in test that loads the real embedding model.

CI excludes it with ``-m "not semantic"`` and ``HF_HUB_OFFLINE=1``, so the model
stack is only exercised on demand:

    pip install -e ".[semantic]"
    GPU_SEARCH_TEST_MODEL=1 pytest tests/test_semantic_model_smoke.py -m semantic
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.semantic

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from gpu_semantic_index import MODEL_ID


@pytest.mark.skipif(
    os.environ.get("GPU_SEARCH_TEST_MODEL") != "1",
    reason="set GPU_SEARCH_TEST_MODEL=1 to load the real embedding model",
)
def test_real_embedding_model_encodes_a_sentence():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_ID, device="cpu")
    vectors = model.encode(["hello world"], normalize_embeddings=True)

    assert vectors.shape[0] == 1
    assert vectors.shape[1] == model.get_sentence_embedding_dimension()
