"""Tests for the shared length-preserving UTF-8 case folding."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gpu_service"))

from gpu_search import TorchByteSearch
from text_fold import PAIR_LUT, fold_bytes, fold_numpy, needs_pair_fold

NON_ASCII_CASES = [
    ("Привет", "привет"),
    ("ПРИВЕТ МИР", "привет мир"),
    ("ÄPFEL", "äpfel"),
    ("Straße", "straße"),
    ("ΣΟΦΟΣ", "σοφοσ"),
    ("µs", "μs"),
    ("mIXed ТеКст", "mixed текст"),
]


@pytest.mark.parametrize("original,expected", NON_ASCII_CASES)
def test_fold_bytes_matches_casefold_for_two_byte_characters(original, expected):
    assert fold_bytes(original.encode("utf-8")) == expected.encode("utf-8")


def test_fold_bytes_matches_bytes_lower_for_ascii():
    for text in ["ABC", "aBc", "Hello World!", "x=1;"]:
        assert fold_bytes(text.encode()) == text.encode().lower()


def test_fold_bytes_preserves_length_and_idempotence():
    for text in ["ПРИВЕТ", "ÄPFEL", "ΣΟΦΟΣ", "mixed Привет", "plain ascii"]:
        raw = text.encode("utf-8")
        folded = fold_bytes(raw)
        assert len(folded) == len(raw)
        assert fold_bytes(folded) == folded


def test_fold_leaves_length_changing_mappings_alone():
    # casefold('ß') == "ss" and casefold('İ') is two characters: folding them
    # would change byte offsets, so they stay untouched on both sides.
    sharp_s = fold_bytes("Straße".encode("utf-8"))
    dotted_i = fold_bytes("İstanbul".encode("utf-8"))
    assert sharp_s == "straße".encode("utf-8")
    assert dotted_i == "İstanbul".encode("utf-8")


def test_fold_numpy_does_not_mutate_input():
    original = np.frombuffer("ПРИВЕТ".encode("utf-8"), dtype=np.uint8).copy()
    snapshot = original.copy()

    folded = fold_numpy(original)

    assert np.array_equal(original, snapshot)
    assert folded is not original
    assert folded.tobytes() == "привет".encode("utf-8")


def test_fold_numpy_handles_empty_and_single_byte_inputs():
    assert fold_numpy(np.array([], dtype=np.uint8)).size == 0
    assert fold_numpy(np.array([ord("A")], dtype=np.uint8)).tolist() == [ord("a")]


def test_pair_lookup_table_is_identity_outside_case_pairs():
    identity = np.arange(65536, dtype=np.uint16)
    changed = PAIR_LUT != identity
    assert changed.any()
    # Every non-identity entry stays inside the two-byte UTF-8 lead range.
    leads = (np.flatnonzero(changed) >> 8).astype(np.uint8)
    assert leads.min() >= 0xC2
    assert leads.max() <= 0xDF


def test_needs_pair_fold_detects_non_ascii():
    assert not needs_pair_fold(b"abc")
    assert not needs_pair_fold(b"abc" * 100)
    assert needs_pair_fold("привет".encode("utf-8"))


def test_torch_fold_matches_numpy_fold_on_cpu():
    searcher = TorchByteSearch(torch.device("cpu"))
    for text in ["ПРИВЕТ", "ÄPFEL", "ΣΟΦΟΣ", "mixed Текст", "ascii only"]:
        raw = text.encode("utf-8")
        tensor = torch.tensor(list(raw), dtype=torch.uint8)
        folded = searcher._fold(tensor).numpy().tobytes()
        assert folded == fold_bytes(raw)


def test_prepared_query_requests_corpus_fold_only_for_non_ascii(monkeypatch):
    searcher = TorchByteSearch(torch.device("cpu"))
    assert searcher.prepare("needle").fold_corpus is False
    assert searcher.prepare("привет").fold_corpus is True
    assert searcher.prepare("привет", case_sensitive=True).fold_corpus is False
