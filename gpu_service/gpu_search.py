"""Storage-agnostic Torch byte-pattern verification."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from text_fold import PAIR_LUT, fold_bytes, needs_pair_fold


@dataclass(frozen=True)
class PreparedQuery:
    encoded: bytes
    pattern: torch.Tensor
    offsets: torch.Tensor
    case_sensitive: bool
    fold_corpus: bool = False

    @property
    def length(self) -> int:
        return len(self.encoded)


class TorchByteSearch:
    """Verify a query against only ``buffer[:valid_length]``.

    This class deliberately has no corpus, file, chunk, or storage imports.  It
    retains the existing first/two-byte prefilter and vectorized full check.

    Case-insensitive matching is ASCII by default, which uses a branch-free
    ``+32`` fold and needs no extra allocation.  Queries containing non-ASCII
    bytes enable ``fold_corpus``: the chunk is folded with the shared
    length-preserving UTF-8 table from :mod:`text_fold` before an exact
    comparison, so Cyrillic/Greek/Latin-1 case differences match as well.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._pair_lut = torch.as_tensor(
            PAIR_LUT.astype(np.int32), dtype=torch.int32, device=device
        )

    def prepare(self, query: str, case_sensitive: bool = False) -> PreparedQuery:
        encoded = query.encode("utf-8", errors="replace")
        fold_corpus = False
        if not case_sensitive:
            encoded = fold_bytes(encoded)
            fold_corpus = needs_pair_fold(encoded)
        return PreparedQuery(
            encoded=encoded,
            pattern=torch.tensor(list(encoded), dtype=torch.uint8, device=self.device),
            offsets=torch.arange(len(encoded), device=self.device),
            case_sensitive=case_sensitive,
            fold_corpus=fold_corpus,
        )

    def search(
        self, buffer: torch.Tensor, valid_length: int, query: PreparedQuery
    ) -> torch.Tensor:
        """Return buffer-relative match starts; no address escapes the caller."""
        if valid_length < 0 or valid_length > buffer.numel():
            raise ValueError("valid_length is outside the supplied buffer")
        match_length = query.length
        if match_length == 0 or valid_length < match_length:
            return torch.empty(0, dtype=torch.long, device=self.device)

        corpus = buffer[:valid_length]
        case_folded = query.fold_corpus and not query.case_sensitive
        if case_folded:
            corpus = self._fold(corpus)

        def equal(values: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
            if query.case_sensitive or case_folded:
                return values == pattern
            return self._ascii_equal(values, pattern)

        limit = valid_length - match_length + 1
        candidates = equal(corpus[:limit], query.pattern[0]).nonzero(
            as_tuple=True
        )[0]
        if len(candidates) == 0:
            return candidates
        if match_length > 1:
            candidates = candidates[equal(corpus[candidates + 1], query.pattern[1])]
            if len(candidates) == 0:
                return candidates
        if match_length > 2:
            indexes = candidates.unsqueeze(1) + query.offsets.unsqueeze(0)
            values = corpus[indexes]
            matches = equal(values, query.pattern.unsqueeze(0)).all(dim=1)
            candidates = candidates[matches]
        return candidates

    def _fold(self, corpus: torch.Tensor) -> torch.Tensor:
        """Return a folded copy of a chunk; length is preserved."""
        ascii_upper = (corpus >= ord("A")) & (corpus <= ord("Z"))
        folded = torch.where(ascii_upper, corpus + 32, corpus)
        if folded.numel() < 2:
            return folded
        codes = (folded[:-1].to(torch.int32) << 8) | folded[1:].to(torch.int32)
        mapped = self._pair_lut[codes]
        lead = folded[:-1]
        mask = (mapped != codes) & (lead >= 0xC2) & (lead <= 0xDF)
        starts = mask.nonzero(as_tuple=True)[0]
        if starts.numel():
            # Changed pairs never start on a continuation byte, so ``starts``
            # and ``starts + 1`` are disjoint and no write clobbers another.
            folded[starts] = (mapped[starts] >> 8).to(torch.uint8)
            folded[starts + 1] = (mapped[starts] & 0xFF).to(torch.uint8)
        return folded

    @staticmethod
    def _ascii_equal(values: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
        equal = values == pattern
        # bytes.lower() maps ASCII A-Z only.  The prepared pattern is lower-case,
        # so accepting its uppercase variant exactly reproduces that behavior
        # without allocating a second lower-cased corpus buffer.
        is_letter = (pattern >= ord("a")) & (pattern <= ord("z"))
        return equal | (is_letter & (values == pattern - 32))
