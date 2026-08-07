"""Storage-agnostic Torch byte-pattern verification."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PreparedQuery:
    encoded: bytes
    pattern: torch.Tensor
    offsets: torch.Tensor
    case_sensitive: bool

    @property
    def length(self) -> int:
        return len(self.encoded)


class TorchByteSearch:
    """Verify a query against only ``buffer[:valid_length]``.

    This class deliberately has no corpus, file, chunk, or storage imports.  It
    retains the existing first/two-byte prefilter and vectorized full check.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def prepare(self, query: str, case_sensitive: bool = False) -> PreparedQuery:
        encoded = query.encode("utf-8", errors="replace")
        if not case_sensitive:
            encoded = encoded.lower()
        return PreparedQuery(
            encoded=encoded,
            pattern=torch.tensor(list(encoded), dtype=torch.uint8, device=self.device),
            offsets=torch.arange(len(encoded), device=self.device),
            case_sensitive=case_sensitive,
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
        limit = valid_length - match_length + 1
        candidates = self._equal(corpus[:limit], query.pattern[0], query.case_sensitive).nonzero(
            as_tuple=True
        )[0]
        if len(candidates) == 0:
            return candidates
        if match_length > 1:
            candidates = candidates[
                self._equal(corpus[candidates + 1], query.pattern[1], query.case_sensitive)
            ]
            if len(candidates) == 0:
                return candidates
        if match_length > 2:
            indexes = candidates.unsqueeze(1) + query.offsets.unsqueeze(0)
            values = corpus[indexes]
            matches = self._equal(values, query.pattern.unsqueeze(0), query.case_sensitive).all(dim=1)
            candidates = candidates[matches]
        return candidates

    @staticmethod
    def _equal(values: torch.Tensor, pattern: torch.Tensor, case_sensitive: bool) -> torch.Tensor:
        equal = values == pattern
        if case_sensitive:
            return equal
        # bytes.lower() maps ASCII A-Z only.  The prepared pattern is lower-case,
        # so accepting its uppercase variant exactly reproduces that behavior
        # without allocating a second lower-cased corpus buffer.
        is_letter = (pattern >= ord("a")) & (pattern <= ord("z"))
        return equal | (is_letter & (values == pattern - 32))
