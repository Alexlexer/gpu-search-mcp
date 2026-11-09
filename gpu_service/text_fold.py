"""Length-preserving UTF-8 case folding shared by search and the trigram index.

The exact-search kernel compares raw bytes, and ``bytes.lower()`` only folds
ASCII ``A-Z``. Repositories with non-English text (Cyrillic, Greek, Latin-1
accents) therefore missed every case-insensitive match: ``привет`` never found
``Привет`` and ``äpfel`` never found ``ÄPFEL``.

This module folds the length-preserving subset of Unicode case mappings: ASCII
plus every codepoint that is encoded as exactly two UTF-8 bytes (U+0080-U+07FF,
which covers Latin-1 Supplement, Latin Extended, Greek and Cyrillic). Folding is
a byte-level, position-independent transform, so a folded corpus can be matched
against a folded query with plain byte equality.

Mappings that change the encoded length (``ß`` -> ``ss``, ``İ`` -> ``i`` plus a
combining dot, the Kelvin sign) are deliberately *not* folded: they cannot be
applied to a byte buffer in place, and skipping them symmetrically keeps exact
and case-insensitive search consistent with each other.

Every consumer must fold with this module. The unpacked trigram postings use the
same transform, otherwise the candidate filter would drop chunks that only match
after folding (a false negative).
"""
from __future__ import annotations

import numpy as np

# U+0080..U+07FF is exactly the set of two-byte UTF-8 codepoints.
_TWO_BYTE_RANGE = range(0x80, 0x800)
_UTF8_LEAD_MIN = 0xC2
_UTF8_LEAD_MAX = 0xDF


def _build_pair_lut() -> np.ndarray:
    """Map every folding two-byte UTF-8 sequence to its case-folded pair."""
    table = np.arange(65536, dtype=np.uint16)
    for codepoint in _TWO_BYTE_RANGE:
        character = chr(codepoint)
        folded_character = character.casefold()
        if len(folded_character) != 1:
            continue
        original = character.encode("utf-8")
        folded = folded_character.encode("utf-8")
        if len(original) != 2 or len(folded) != 2 or original == folded:
            continue
        table[(original[0] << 8) | original[1]] = (folded[0] << 8) | folded[1]
    return table


PAIR_LUT: np.ndarray = _build_pair_lut()


def fold_bytes(data: bytes) -> bytes:
    """Return a copy of ``data`` with its case folded, preserving length."""
    return fold_numpy(np.frombuffer(data, dtype=np.uint8)).tobytes()


def fold_numpy(values: np.ndarray) -> np.ndarray:
    """Fold ASCII and two-byte UTF-8 upper-case letters in a byte array.

    The returned array has the same length as ``values``; the input is never
    modified.
    """
    folded = values.astype(np.uint8, copy=True)
    if not folded.size:
        return folded
    ascii_upper = (folded >= ord("A")) & (folded <= ord("Z"))
    folded[ascii_upper] += 32
    if folded.size < 2:
        return folded
    codes = (folded[:-1].astype(np.uint16) << 8) | folded[1:].astype(np.uint16)
    mapped = PAIR_LUT[codes]
    lead = folded[:-1]
    mask = (
        (mapped != codes)
        & (lead >= _UTF8_LEAD_MIN)
        & (lead <= _UTF8_LEAD_MAX)
    )
    starts = np.flatnonzero(mask)
    if not starts.size:
        return folded
    # Changed pairs never start on a continuation byte, so ``starts`` and
    # ``starts + 1`` are disjoint and no write can clobber another.
    folded[starts] = (mapped[starts] >> 8).astype(np.uint8)
    folded[starts + 1] = (mapped[starts] & 0xFF).astype(np.uint8)
    return folded


def needs_pair_fold(data: bytes) -> bool:
    """True when ``data`` contains any non-ASCII byte that could be folded."""
    return any(byte >= 0x80 for byte in data)
