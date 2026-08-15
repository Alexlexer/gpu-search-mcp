"""Storage-agnostic Torch byte-pattern verification."""
from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_MAX_QUERY_BYTES = 256 * 1024
DEFAULT_MATCH_WORKSPACE_BYTES = 16 * 1024 * 1024
_MATCH_WORKSPACE_BYTES_PER_ELEMENT = 16


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

    def __init__(
        self,
        device: torch.device,
        *,
        max_query_bytes: int = DEFAULT_MAX_QUERY_BYTES,
        match_workspace_bytes: int = DEFAULT_MATCH_WORKSPACE_BYTES,
    ):
        if max_query_bytes <= 0:
            raise ValueError("max_query_bytes must be positive")
        if match_workspace_bytes <= 0:
            raise ValueError("match_workspace_bytes must be positive")
        self.device = device
        self.max_query_bytes = max_query_bytes
        self.match_workspace_bytes = match_workspace_bytes

    def prepare(self, query: str, case_sensitive: bool = False) -> PreparedQuery:
        encoded = query.encode("utf-8", errors="replace")
        if len(encoded) > self.max_query_bytes:
            raise ValueError(
                f"query is {len(encoded)} bytes; maximum is {self.max_query_bytes} bytes"
            )
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
            # The fully vectorized candidates x query-length matrix can be enormous
            # for dense inputs. Verify fixed-size candidate batches so temporary
            # tensors stay within a predictable per-search workspace.
            bytes_per_candidate = max(
                1, match_length * _MATCH_WORKSPACE_BYTES_PER_ELEMENT
            )
            batch_size = max(1, self.match_workspace_bytes // bytes_per_candidate)
            verified: list[torch.Tensor] = []
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start:start + batch_size]
                indexes = batch.unsqueeze(1) + query.offsets.unsqueeze(0)
                values = corpus[indexes]
                matches = self._equal(
                    values, query.pattern.unsqueeze(0), query.case_sensitive
                ).all(dim=1)
                verified.append(batch[matches])
            candidates = torch.cat(verified) if verified else candidates[:0]
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
